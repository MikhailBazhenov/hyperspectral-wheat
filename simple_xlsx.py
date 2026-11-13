"""Dependency-free XLSX output for pandas DataFrames.

Single worksheet ("Data"), ordinary cells, no styling, no formulas.
Uses only Python's standard library; preserves numerical/boolean cell types and
writes non-finite numeric values as empty cells. Excel and LibreOffice compatible.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import tempfile
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

_XLSX_ROW_LIMIT = 1_048_576
_XLSX_COL_LIMIT = 16_384
_XML_INVALID = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]')


def _excel_column(index):
    """0-based numeric index to Excel column letters."""
    index += 1
    parts = []
    while index:
        index, remainder = divmod(index - 1, 26)
        parts.append(chr(65 + remainder))
    return ''.join(reversed(parts))


def _cell_xml(value, reference):
    if value is None:
        return ''
    # NumPy scalar support without depending on NumPy when used separately.
    if hasattr(value, 'item') and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError, AttributeError):
            pass
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(value):
            return ''
        return f'<c r="{reference}"><v>{value}</v></c>'
    # Text is ALWAYS text: labels like '=x' must not be interpreted as formulas.
    value = str(value)
    if not value:
        return ''
    value = _XML_INVALID.sub('', value)[:32767]
    return f'<c r="{reference}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'


def write_dataframe_xlsx(frame, path, *, index=False, index_label=None):
    """Write a DataFrame to a minimal, Excel-readable XLSX workbook.

    Parameters mirror DataFrame.to_csv for the index/index_label options.
    Intentional: no style/colours/charts, and no optional XLSX package required.
    Raises on Excel's row/column limits instead of silently dropping data.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if index:
        frame = frame.reset_index()
        if index_label is not None:
            frame = frame.rename(columns={frame.columns[0]: index_label})
    columns = [str(x) for x in frame.columns]
    nr, nc = len(frame) + 1, len(columns)
    if nc > _XLSX_COL_LIMIT or nr > _XLSX_ROW_LIMIT:
        raise ValueError(f'Excel limit exceeded: {nr} rows x {nc} columns')
    if nc == 0:
        raise ValueError('Cannot export DataFrame with zero columns')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(prefix='._xlsx_', suffix='.xlsx', dir=dest.parent, delete=False) as stream:
            tmp = Path(stream.name)
        with ZipFile(tmp, 'w', compression=ZIP_DEFLATED, compresslevel=5) as archive:
            archive.writestr('[Content_Types].xml',
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                '</Types>')
            archive.writestr('_rels/.rels',
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                '</Relationships>')
            archive.writestr('xl/workbook.xml',
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
                ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '<sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>')
            archive.writestr('xl/_rels/workbook.xml.rels',
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                '</Relationships>')
            archive.writestr('xl/styles.xml',
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
                '<fills count="2"><fill><patternFill patternType="none"/></fill>'
                '<fill><patternFill patternType="gray125"/></fill></fills>'
                '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
                '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
                '</styleSheet>')
            with archive.open('xl/worksheets/sheet1.xml', 'w', force_zip64=True) as stream:
                def put(s): stream.write(s.encode('utf-8'))
                last = f'{_excel_column(nc-1)}{nr}'
                put('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                    f'<dimension ref="A1:{last}"/>'
                    '<sheetViews><sheetView workbookViewId="0">'
                    '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
                    '</sheetView></sheetViews><sheetData>')
                put('<row r="1">')
                for c, value in enumerate(columns):
                    put(_cell_xml(value, f'{_excel_column(c)}1'))
                put('</row>')
                for r, vals in enumerate(frame.itertuples(index=False, name=None), start=2):
                    put(f'<row r="{r}">')
                    for c, value in enumerate(vals):
                        cell = _cell_xml(value, f'{_excel_column(c)}{r}')
                        if cell:
                            put(cell)
                    put('</row>')
                put('</sheetData>')
                # Excel filter to help sort without making a styled table.
                put(f'<autoFilter ref="A1:{last}"/></worksheet>')
        os.replace(tmp, dest)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    return dest


def read_dataframe_xlsx(path):
    """Read this pipeline's simple one-sheet XLSX using stdlib (no Excel engine).

    Supports normal numeric, boolean, inlineStr and shared-string cells. This
    reader is for pipeline-generated workbooks, not arbitrary styled Excel docs.
    """
    import pandas as pd
    from xml.etree import ElementTree as ET
    from zipfile import ZipFile
    main_ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with ZipFile(path) as archive:
        shared=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            root=ET.fromstring(archive.read('xl/sharedStrings.xml'))
            for si in root.findall(main_ns+'si'):
                shared.append(''.join(node.text or '' for node in si.iter(main_ns+'t')))
        root=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        rows=[]
        import re
        def col_index(ref):
            letters=re.match(r'[A-Z]+',ref).group()
            col=0
            for letter in letters:col=col*26+(ord(letter)-64)
            return col-1
        for row in root.iter(main_ns+'row'):
            values={}
            for cell in row.findall(main_ns+'c'):
                idx=col_index(cell.attrib['r'])
                t=cell.attrib.get('t','n')
                v=cell.find(main_ns+'v')
                if t=='inlineStr':
                    item=cell.find(main_ns+'is')
                    value=''.join(node.text or '' for node in item.iter(main_ns+'t')) if item is not None else ''
                elif v is None or v.text is None:value=None
                elif t=='s':value=shared[int(v.text)]
                elif t=='b':value=v.text=='1'
                elif t=='n':
                    try:value=float(v.text)
                    except ValueError:value=None
                else:value=v.text
                values[idx]=value
            rows.append(values)
    if not rows:return pd.DataFrame()
    nc=max(rows[0],default=-1)+1
    header=[str(rows[0].get(i,'')) for i in range(nc)]
    return pd.DataFrame([[row.get(i) for i in range(nc)] for row in rows[1:]],columns=header)
