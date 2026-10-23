"""Do not mutate a user's environment or download tracker packages at runtime."""
import os
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ.setdefault('MPLBACKEND', 'Agg')
