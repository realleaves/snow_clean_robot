"""Export reproducible mock evidence and source review drafts, not a filing."""
import argparse
import hashlib
import html
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from scenario_runner import SCENARIOS, run_scenario

ROOT = Path(__file__).resolve().parent.parent
SOFTWARE_NAME = '室内雪迹清除机器人多模态感知与智能决策系统'
VERSION = 'V1.0'
SOURCE_DIRS = ('camera', 'perception', 'fusion', 'mapping', 'planning',
               'decision', 'interface', 'feedback', 'visualization', 'utils', 'tools')


def collect_sources(root: Path = ROOT) -> list[Path]:
    paths = [root / 'main.py', root / 'system.py', root / 'scenario_runner.py']
    for directory in SOURCE_DIRS:
        paths.extend(sorted((root / directory).glob('*.py')))
    return [path for path in paths if path.read_text(encoding='utf-8').strip()]


def export_sources(destination: Path) -> dict:
    records = []
    source_lines = []
    for path in collect_sources():
        content = path.read_text(encoding='utf-8')
        relative = path.relative_to(ROOT).as_posix()
        lines = content.splitlines()
        start = len(source_lines) + 1
        source_lines.extend(lines)
        records.append(dict(path=relative, lines=len(lines),
                            combined_start_line=start, combined_end_line=len(source_lines),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (destination / 'source_full.txt').write_text('\n'.join(source_lines) + '\n', encoding='utf-8')
    printable = [(record['path'], number, line)
                 for record in records
                 for number, line in enumerate(
                     (ROOT / record['path']).read_text(encoding='utf-8').splitlines(), 1)
                 if line.strip()]
    pages = [printable[offset:offset + 50] for offset in range(0, len(printable), 50)]
    selected = list(range(len(pages))) if len(pages) <= 60 else list(range(30)) + list(range(len(pages) - 30, len(pages)))
    sections = []
    for index, original in enumerate(selected, 1):
        rows = '\n'.join(f'{html.escape(path)}:{number:04d}  {html.escape(line)}'
                         for path, number, line in pages[original])
        sections.append(f'<section><header>{SOFTWARE_NAME} {VERSION} — 源程序整理稿'
                        f'<span>{index}/{len(selected)}（原序第 {original + 1} 页）</span></header>'
                        f'<pre>{rows}</pre></section>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<title>源程序整理稿</title><style>'
                '@page{size:A4 landscape;margin:10mm}body{margin:0;color:#111}'
                'section{break-after:page}section:last-child{break-after:auto}'
                'header{font:12px sans-serif;border-bottom:1px solid #888;padding:6px 0}'
                'span{float:right}pre{font:8px/1.4 monospace;white-space:pre;tab-size:4}'
                '</style>' + ''.join(sections) + '</html>')
    (destination / 'source_review.html').write_text(document, encoding='utf-8')
    return dict(files=records, total_lines=len(source_lines), nonblank_lines=len(printable),
                full_pages=len(pages), selected_pages=len(selected),
                note='空行仅在打印整理稿中省略；原始完整内容见 source_full.txt。'
                     '这是待人工检查排版的整理稿，不是已审核合格的提交件。')


def write_gallery(destination: Path, reports: list[dict]) -> None:
    sections = []
    for report in reports:
        checks = ''.join(f'<li>{"通过" if passed else "失败"}：{html.escape(name)}</li>'
                         for name, passed in report['checks'].items())
        images = ''.join(f'<figure><img src="screenshots/{html.escape(name)}">'
                         f'<figcaption>{html.escape(name)}</figcaption></figure>'
                         for name in report['screenshots'])
        sections.append(f'<section><h2>{report["key"]} {html.escape(report["title"])}</h2>'
                        f'<p>{html.escape(report["description"])}</p><ul>{checks}</ul>{images}</section>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<title>V1.0 虚拟运行证据</title><style>'
                'body{font:16px sans-serif;max-width:1280px;margin:32px auto;padding:16px}'
                'img{width:100%;border:1px solid #ccc}figure{margin:24px 0}'
                'figcaption{color:#555}section{border-top:2px solid #ddd;margin-top:36px}'
                '</style><h1>' + SOFTWARE_NAME + ' ' + VERSION + '</h1>'
                '<p>以下为本次实际 Mock 运行生成的监控面板，不是真机照片。'
                '图像为最近一次采集帧，状态为当前执行状态；无新帧的中间状态复用最近图像。'
                'G/H 为预期故障注入，ERROR 与安全停机属于相应场景的预期结果。'
                '场景分级使用确定性提示值，不代表真实识别准确率。</p>'
                + ''.join(sections) + '</html>')
    (destination / 'index.html').write_text(document, encoding='utf-8')


def export_materials(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    reports = [run_scenario(key, destination / 'screenshots') for key in sorted(SCENARIOS)]
    report = dict(software_name=SOFTWARE_NAME, version=VERSION,
                  generated_at=datetime.now(timezone.utc).isoformat(), mode='mock',
                  environment=dict(python=platform.python_version(), system=platform.system(),
                                   opencv=cv2.__version__, numpy=np.__version__, pyyaml=yaml.__version__),
                  passed=all(item['passed'] for item in reports), scenarios=reports)
    (destination / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    source_manifest = export_sources(destination)
    write_gallery(destination, reports)
    for name in ('USER_MANUAL.md', 'COPYRIGHT_CHECKLIST.md'):
        shutil.copyfile(ROOT / 'docs' / name, destination / name)
    manifest = dict(software_name=SOFTWARE_NAME, version=VERSION, sources=source_manifest,
                    artifacts={path.relative_to(destination).as_posix():
                               hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(destination.rglob('*')) if path.is_file()})
    (destination / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description='导出软著准备材料（虚拟运行证据与待核对整理稿）')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' /
                        datetime.now().strftime('copyright-%Y%m%d-%H%M%S-%f'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output directory already exists; choose a new path to preserve earlier evidence')
    try:
        report = export_materials(args.output)
    except (OSError, ValueError) as exc:
        print(f'材料导出失败（可能留下未完成目录，请勿提交）: {exc}')
        return 1
    print(f'材料目录: {args.output.resolve()}')
    print(f'虚拟场景: {sum(item["passed"] for item in report["scenarios"])}/8 通过')
    print('先打开 index.html 检查截图，再核对 COPYRIGHT_CHECKLIST.md；此工具不会提交申请。')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
