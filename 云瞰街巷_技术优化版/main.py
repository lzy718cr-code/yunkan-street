from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description='云瞰街巷交通观察工作台')
    parser.add_argument('--smoke-test',metavar='IMAGE_OR_VIDEO')
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--max-frames',type=int,default=12)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--model',type=Path,help='覆盖默认权重，便于同一程序切换车辆/空间要素模型')
    parser.add_argument('--mode',choices=['vehicle','person'],help='场景模式：车辆观测或人行观测')
    parser.add_argument('--imgsz',type=int,choices=[416,640,960,1024,1280])
    args=parser.parse_args()
    # A packaged windowed app has no console. Keep library diagnostics usable.
    data_root=args.data_root
    if data_root is not None:
        data_root.mkdir(parents=True,exist_ok=True)
    for stream_name in ('stdout','stderr'):
        if getattr(sys,stream_name) is None:
            setattr(sys,stream_name,open(os.devnull,'w',encoding='utf-8'))
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFontDatabase, QFont
    from PySide6.QtWidgets import QApplication
    from traffic_app.ui import MainWindow, THEME, default_data_root
    data_root=args.data_root or default_data_root();data_root.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=str(data_root/'application.log'),level=logging.INFO,encoding='utf-8',format='%(asctime)s %(levelname)s %(message)s')
    app=QApplication(sys.argv[:1])
    if not QFontDatabase.families():
        font_path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/'msyh.ttc'
        if font_path.is_file():QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont('Microsoft YaHei UI',9))
    app.setApplicationName('云瞰街巷');app.setStyleSheet(THEME)
    window=MainWindow(data_root);window.show()
    if args.mode is not None:
        index=window.mode.findData(args.mode)
        if index>=0:window.mode.setCurrentIndex(index)
    if args.model is not None:
        window.model_path=args.model
    if args.imgsz is not None:
        index=window.imgsz.findData(args.imgsz)
        if index>=0:window.imgsz.setCurrentIndex(index)
    outcome={'exit_code':0}
    if args.smoke_test:
        index=window.device.findData(args.device)
        if index>=0:window.device.setCurrentIndex(index)
        window.test_max_frames=args.max_frames
        window.suppress_dialogs=True
        window.set_source(str(Path(args.smoke_test).resolve()))
        def completed(summary):
            outcome['exit_code']=0 if summary.get('status') in ['completed','limited'] else 1
            if args.report:
                args.report.parent.mkdir(parents=True,exist_ok=True)
                args.report.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
            state={'waits':0}
            def finish():
                window.poll_frame()
                running=window.worker is not None and window.worker.isRunning()
                portrait=getattr(window,'portrait_worker',None)
                if running or (portrait is not None and portrait.isRunning()):
                    QTimer.singleShot(150,finish);return
                # The portrait is queued when the session ends; give that turn
                # of the event loop a chance before deciding to close.
                if (portrait is None and getattr(window,'last_session_dir',None) is not None
                        and state['waits'] < 20):
                    state['waits']+=1;QTimer.singleShot(150,finish);return
                if args.screenshot:
                    args.screenshot.parent.mkdir(parents=True,exist_ok=True);window.grab().save(str(args.screenshot))
                window.close();app.quit()
            QTimer.singleShot(300,finish)
        window.session_completed.connect(completed)
        QTimer.singleShot(350,window.start_analysis)
        def timeout():
            outcome['exit_code']=2
            if window.worker is not None:window.worker.stop()
            window.close()
        # Long recordings need more than the original three-minute smoke-test cap.
        QTimer.singleShot(3600000,timeout)
    app.exec()
    return outcome['exit_code']


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
