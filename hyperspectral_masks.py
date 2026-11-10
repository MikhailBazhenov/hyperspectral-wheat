"""Reusable object masks and an interactive Jupyter mask editor (SAM + brush).

Masks are stored as SAMPLE/masks/<object_id>.png, 8-bit 0/255. When the
masks directory is absent, legacy SAMPLE/wheat_mask.png is read as object_01.
This module never silently overwrites a legacy wheat mask.
"""
from pathlib import Path
from functools import lru_cache
import re
import numpy as np
import cv2
from PIL import Image
from spectral_cube_reflectance import load_quant_cube

DEFAULT_SAM_ID='facebook/sam-vit-base'

def _validate_id(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,70}',str(name)):
        raise ValueError('Object ID must use Latin letters, digits, underscore or hyphen')
    return str(name)

def list_object_masks(sample_folder, *, include_legacy=True):
    sample=Path(sample_folder)
    found={p.stem:p for p in sorted((sample/'masks').glob('*.png')) if p.is_file()} if (sample/'masks').is_dir() else {}
    if include_legacy and not found and (sample/'wheat_mask.png').is_file():
        found={'object_01':sample/'wheat_mask.png'}
    return found

def save_object_mask(sample_folder, object_id, mask, *, overwrite=True):
    name=_validate_id(object_id)
    sample=Path(sample_folder)
    cube_file=sample/'corrected_images'/'corrected_cube.npz'
    if not cube_file.is_file():
        raise FileNotFoundError(cube_file)
    # Read only the NPY header inside NPZ; never decompress the multi-GB cube just to save a mask.
    import zipfile
    with zipfile.ZipFile(cube_file) as z, z.open('cube.npy') as stream:
        magic=np.lib.format.read_magic(stream)
        if magic==(1,0):
            shape,_,_=np.lib.format.read_array_header_1_0(stream)
        else:
            shape,_,_=np.lib.format.read_array_header_2_0(stream)
        shape=shape[:2]
    a=np.asarray(mask)
    if a.shape != shape or a.ndim != 2:
        raise ValueError(f'Mask shape {a.shape} must match cube {shape}')
    dest=sample/'masks'/f'{name}.png'
    dest.parent.mkdir(exist_ok=True,parents=True)
    if dest.exists() and not overwrite:
        raise FileExistsError(dest)
    tmp=dest.with_name(dest.stem+'.tmp.png')
    if not cv2.imwrite(str(tmp),np.where(a,255,0).astype('uint8')):
        raise IOError(f'Cannot write mask {tmp}')
    tmp.replace(dest)
    return dest

def read_mask(path, expected_shape):
    a=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if a is None or a.shape != expected_shape:
        raise ValueError(f'Mask missing or wrong size: {path}, required={expected_shape}')
    return a>0

def make_rgb_from_cube(cube_path, rgb_wavelengths=(630,530,470),percentiles=(1,99)):
    """RGB from corrected preview bands; avoids loading the full NPZ for SAM."""
    from segment_wheat_gwfss import _load_band,_to_uint8
    folder=Path(cube_path).parent
    bands=[]
    for nm in rgb_wavelengths:
        try:
            bands.append(_to_uint8(_load_band(folder,int(nm)),stretch_percentiles=percentiles))
        except ValueError as exc:
            # Fallback if an older generated cube has no preview images.
            if 'Expected one image' not in str(exc):raise
            break
    if len(bands)==3 and len({b.shape for b in bands})==1:
        return np.stack(bands,axis=-1)
    cube,wls,valid,_=load_quant_cube(cube_path)
    parts=[]
    for wl in rgb_wavelengths:
        k=int(np.argmin(np.abs(wls-int(wl))))
        band=np.where(valid[:,:,k],cube[:,:,k],np.nan)
        f=band[np.isfinite(band)]
        if not len(f):
            parts.append(np.zeros(cube.shape[:2],dtype=np.uint8));continue
        lo,hi=np.percentile(f,percentiles)
        hi=max(float(hi),float(lo)+1e-8)
        parts.append(np.nan_to_num(np.clip((band-lo)/(hi-lo),0,1)*255,nan=0).astype(np.uint8))
    return np.stack(parts,axis=2)

def _sam_folder(model_id):
    name='SAM_'+re.sub(r'[^A-Za-z0-9_-]+','_',model_id)
    return Path(__file__).resolve().parent/name

def _sam_ready(p):
    return (p/'config.json').is_file() and (p/'preprocessor_config.json').is_file() and any((p/n).is_file() for n in ('model.safetensors','pytorch_model.bin'))

@lru_cache(maxsize=2)
def _load_sam(model_id,device):
    import torch
    from transformers import SamModel,SamProcessor
    folder=_sam_folder(model_id)
    if not _sam_ready(folder):
        from huggingface_hub import snapshot_download
        print(f'SAM model not found locally; downloading once to {folder}')
        folder.mkdir(exist_ok=True,parents=True)
        snapshot_download(repo_id=model_id,local_dir=str(folder),
                          allow_patterns=['*.json','*.safetensors','pytorch_model*.bin'])
        if not _sam_ready(folder):
            raise RuntimeError(f'Incomplete SAM model folder: {folder}')
    proc=SamProcessor.from_pretrained(str(folder),local_files_only=True)
    model=SamModel.from_pretrained(str(folder),local_files_only=True).to(device).eval()
    return proc,model

def predict_sam_mask(rgb,points,labels,*,model_id=DEFAULT_SAM_ID,device=None):
    """Positive/negative clicks in original image pixel coordinates."""
    if not points or not any(x==1 for x in labels):
        raise ValueError('At least one foreground (+) point is required for SAM')
    import torch
    if device is None:
        device='cuda' if torch.cuda.is_available() else 'cpu'
    proc,model=_load_sam(model_id,str(device))
    inputs=proc(images=Image.fromarray(np.asarray(rgb,dtype=np.uint8)),
                input_points=[[list(map(float,p)) for p in points]],
                input_labels=[list(map(int,labels))],return_tensors='pt')
    moved={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in inputs.items()}
    with torch.inference_mode():
        result=model(**moved,multimask_output=True)
    processed=proc.image_processor.post_process_masks(result.pred_masks.cpu(),
        inputs['original_sizes'].cpu(),inputs['reshaped_input_sizes'].cpu())
    scores=result.iou_scores[0,0].detach().cpu().numpy().reshape(-1)
    choice=int(np.argmax(scores))
    a=processed[0][0,choice].detach().cpu().numpy()
    return np.asarray(a>0,dtype=bool),float(scores[choice])

class MaskEditor:
    """Interactive ipympl editor. No file is changed until Save is clicked.

    Modes: positive/negative point, brush, eraser. Drag with mouse to paint.
    Undo restores the last brush stroke or SAM proposal; Undo point removes
    the last point. New object starts an empty mask with a new ID.
    """
    def __init__(self,sample_folder,object_id=None,*,rgb_wavelengths=(630,530,470),
                 sam_model_id=DEFAULT_SAM_ID,brush_radius=16):
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display
        self.sample=Path(sample_folder)
        self.rgb=make_rgb_from_cube(self.sample/'corrected_images'/'corrected_cube.npz',rgb_wavelengths)
        self.h,self.w=self.rgb.shape[:2]
        self.sam_model_id=sam_model_id
        self.available=list_object_masks(self.sample)
        self.object_id=object_id or (next(iter(self.available)) if self.available else 'object_01')
        self._load_current()
        self.points=[];self.labels=[];self.undo_stack=[]
        self.dragging=False;self.drag_original=None
        self.mode=widgets.ToggleButtons(options=[('Positive +','positive'),('Negative −','negative'),
            ('Brush','brush'),('Eraser','eraser')],value='positive',description='Tool:')
        self.radius=widgets.IntSlider(value=brush_radius,min=1,max=100,description='Radius')
        self.name=widgets.Text(value=self.object_id,description='Object ID')
        self.select=widgets.Dropdown(options=[('New object','__new__')]+[(x,x) for x in self.available],
                                      value=self.object_id if self.object_id in self.available else '__new__',description='Existing')
        self.status=widgets.HTML(value='<b>Click to place points. Save writes a new mask file.</b>')
        buttons={}
        for label,key in [('Run SAM','sam'),('Undo point','undo_point'),('Clear points','clear_points'),
                          ('Undo edit','undo'),('Clear mask','clear'),('Save mask','save'),
                          ('New object','new'),('Skip / close','skip'),('Cancel without saving','cancel')]:
            b=widgets.Button(description=label,layout=widgets.Layout(width='145px'))
            b.on_click(lambda _,key=key:self._button(key));buttons[key]=b
        self.fig,self.ax=plt.subplots(figsize=(12,7))
        self.ax.set_axis_off();self.ax.set_title(self.sample.name)
        self.base_artist=self.ax.imshow(self.rgb)
        self.overlay=self.ax.imshow(np.zeros((self.h,self.w,4),dtype=np.uint8),interpolation='nearest')
        self.marker_plot=self.ax.scatter([],[],s=26)
        self.fig.canvas.mpl_connect('button_press_event',self._press)
        self.fig.canvas.mpl_connect('motion_notify_event',self._move)
        self.fig.canvas.mpl_connect('button_release_event',self._release)
        self._refresh()
        self.widget=widgets.VBox([
            widgets.HBox([self.select,self.name]),
            widgets.HBox([self.mode,self.radius]),
            widgets.HBox([buttons[k] for k in ['sam','undo_point','clear_points','undo']]),
            widgets.HBox([buttons[k] for k in ['clear','save','new','skip','cancel']]),
            self.status,self.fig.canvas])
        self.select.observe(self._select_change,names='value')
        display(self.widget)

    def _load_current(self):
        p=self.available.get(self.object_id)
        self.mask=read_mask(p,(self.h,self.w)) if p else np.zeros((self.h,self.w),dtype=bool)
    def _select_change(self,change):
        if change['new']=='__new__':return
        self.object_id=change['new'];self.name.value=self.object_id
        self._load_current();self.points=[];self.labels=[];self.undo_stack=[]
        self._refresh();self._message('Loaded mask; unsaved edits in previous object discarded.')
    def _message(self,txt):self.status.value=txt
    def _refresh(self):
        rgba=np.zeros((self.h,self.w,4),dtype=np.uint8)
        rgba[self.mask]=[50,255,60,110]
        self.overlay.set_data(rgba)
        if self.points:
            self.marker_plot.set_offsets(np.array(self.points))
            self.marker_plot.set_color(['lime' if l==1 else 'red' for l in self.labels])
        else:self.marker_plot.set_offsets(np.empty((0,2)))
        self.fig.canvas.draw_idle()
    def _pixel(self,event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:return None
        return int(np.clip(round(event.xdata),0,self.w-1)),int(np.clip(round(event.ydata),0,self.h-1))
    def _paint(self,x,y):
        r=int(self.radius.value)
        x0=max(0,x-r);x1=min(self.w,x+r+1);y0=max(0,y-r);y1=min(self.h,y+r+1)
        yy,xx=np.ogrid[y0:y1,x0:x1]
        self.mask[y0:y1,x0:x1][(xx-x)**2+(yy-y)**2<=r*r]=(self.mode.value=='brush')
        self._refresh()
    def _press(self,event):
        if getattr(event,'button',None)!=1:return
        point=self._pixel(event)
        if point is None:return
        if self.mode.value in ('brush','eraser'):
            self.dragging=True;self.drag_original=self.mask.copy();self._paint(*point)
        else:
            self.points.append(point);self.labels.append(1 if self.mode.value=='positive' else 0)
            self._refresh()
    def _move(self,event):
        if self.dragging and self.mode.value in ('brush','eraser'):
            point=self._pixel(event)
            if point is not None:self._paint(*point)
    def _release(self,event):
        if self.dragging:
            self.undo_stack.append(self.drag_original)
            self.undo_stack=self.undo_stack[-8:]
            self.dragging=False;self.drag_original=None
    def _button(self,key):
        try:
            if key=='undo_point':
                if self.points:self.points.pop();self.labels.pop()
            elif key=='clear_points':self.points=[];self.labels=[]
            elif key=='undo':
                if self.undo_stack:self.mask=self.undo_stack.pop()
            elif key=='clear':self.undo_stack.append(self.mask.copy());self.mask[:]=False
            elif key=='sam':
                predicted,score=predict_sam_mask(self.rgb,self.points,self.labels,model_id=self.sam_model_id)
                self.undo_stack.append(self.mask.copy());self.mask=predicted
                self._message(f'SAM candidate (predicted IoU {score:.3f}); refine with brush/eraser, then save.')
            elif key=='save':
                path=save_object_mask(self.sample,self.name.value,self.mask)
                self.object_id=self.name.value
                self.available=list_object_masks(self.sample)
                self.select.options=[('New object','__new__')]+[(x,x) for x in self.available]
                self._message(f'<b>Saved: {path}</b> ({int(self.mask.sum())} pixels).')
            elif key=='new':
                self.available=list_object_masks(self.sample)
                ids=set(self.available)
                n=1
                while f'object_{n:02d}' in ids:n+=1
                self.object_id=f'object_{n:02d}';self.name.value=self.object_id
                self.mask=np.zeros((self.h,self.w),bool);self.points=[];self.labels=[];self.undo_stack=[]
                self.select.value='__new__'
                self._message('New empty object; old file unchanged unless Save was clicked.')
            elif key in ('skip','cancel'):
                import matplotlib.pyplot as plt
                plt.close(self.fig);self.widget.close()
                return
            self._refresh()
        except Exception as ex:
            self._message(f'<b style="color:red">{type(ex).__name__}: {ex}</b>')

def edit_mask(sample_folder, object_id=None, **kwargs):
    """Run in a notebook after `%matplotlib widget`; returns MaskEditor object."""
    return MaskEditor(sample_folder,object_id=object_id,**kwargs)

def segment_samples(main_folder, *, model='segformer',overwrite=False,
                    rgb_wavelengths=(630,530,470),**segformer_options):
    """Generate initial SegFormer masks, or provide SAM/manual editor entry points.

    SAM is interactive; call edit_mask(sample) for each desired sample rather
    than creating many simultaneous UI windows here.
    """
    from segment_wheat_gwfss_tiled import segment_wheat_gwfss_tiled
    main=Path(main_folder)
    rows=[]
    if model not in ('segformer','sam','none'):
        raise ValueError('model must be segformer, sam or none')
    for cube in sorted(main.rglob('corrected_cube.npz')):
        sample=cube.parent.parent
        if model=='segformer':
            path=sample/'masks'/'object_01.png'
            if not path.is_file() and not overwrite and (sample/'wheat_mask.png').is_file():
                # Existing v5 studies: preserve the already accepted mask instead
                # of silently replacing it with fresh neural inference.
                legacy=cv2.imread(str(sample/'wheat_mask.png'),cv2.IMREAD_GRAYSCALE)
                if legacy is None:raise OSError(f'Cannot read legacy mask in {sample}')
                save_object_mask(sample,'object_01',legacy>0,overwrite=False)
                print(f'{sample.name}: imported legacy wheat_mask.png -> masks/object_01.png')
            elif not path.is_file() or overwrite:
                path.parent.mkdir(parents=True,exist_ok=True)
                segment_wheat_gwfss_tiled(images_folder=cube.parent,mask_path=path,
                    rgb_wavelengths=rgb_wavelengths,**segformer_options)
            rows.append((sample.name,str(path),'saved'))
        else:
            rows.append((sample.name,'',f'Use edit_mask({str(sample)!r}) for interactive SAM/manual segmentation'))
    return rows
