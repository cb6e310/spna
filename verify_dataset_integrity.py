import os
import sys
import zipfile
import zlib

base = '/home/song/projects/trust_align/01_data_protocol/datasets'
for version in ('1.0', '2.0'):
    archive = f'{base}/_downloads/zuco{version}-osfstorage.zip'
    root = f'{base}/zuco_{version}'
    print('VERIFY', version, flush=True)
    count = total = 0
    issues = []
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            path = os.path.join(root, item.filename)
            if not os.path.isfile(path) or os.path.getsize(path) != item.file_size:
                issues.append((item.filename, 'missing_or_size'))
                continue
            crc = 0
            with open(path, 'rb') as f:
                while chunk := f.read(8 * 1024 * 1024):
                    crc = zlib.crc32(chunk, crc)
            if crc & 0xffffffff != item.CRC:
                issues.append((item.filename, 'crc'))
            count += 1
            total += item.file_size
            if count % 100 == 0:
                print('progress', version, count, round(total / 2**30, 2), 'GiB', flush=True)
    print('RESULT', version, 'files', count, 'GiB', round(total / 2**30, 2), 'issues', len(issues), flush=True)
    for issue in issues:
        print('ISSUE', repr(issue), flush=True)
    if issues:
        sys.exit(1)
