import sys
import os
import argparse
import time
import re
import numpy as np
import json
import yt_dlp
import argparse
import multiprocessing
from functools import partial
from contextlib import contextmanager
from tqdm import tqdm

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def update_progress(*a):
    pbar.update()

def download_video(url, fmt, video_dir):
    cmd = ['yt-dlp', '--ignore-errors', 
           '--download-archive', 'scraping/downloaded_video.txt', 
           '--format', fmt, 
           '-o', '"{}/%(id)s.video.%(ext)s"'.format(video_dir),
           '"{}"'.format(url)]
    os.system(' '.join(cmd))


def download_audio(url, fmt, audio_dir):
    cmd = ['yt-dlp', '--ignore-errors', 
           '--download-archive', 'scraping/downloaded_audio.txt', 
           '--format', fmt, 
           '-o', '"{}/%(id)s.audio.f%(format_id)s.%(ext)s"'.format(audio_dir),
           '"{}"'.format(url)]
    os.system(' '.join(cmd))

def download_video_audio(args):
    yid, video_fmt, audio_fmt, output_dir = args
    if yid not in video_fmt or yid not in audio_fmt:
        return
    url = 'https://www.youtube.com/watch?v=' + yid
    
    download_video(url, video_fmt[yid], output_dir)
    download_audio(url, audio_fmt[yid], output_dir)
    time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_list', default='meta/spatialaudiogen_db.lst', help='File containing list of youtube ids.')
    parser.add_argument('--output_dir', default='data/orig', help='Output folder.')
    parser.add_argument('--low_res', action='store_true', help='Download low resolution videos.')
    args = parser.parse_args(sys.argv[1:])

    youtube_ids = open(args.db_list).read().splitlines()
    audio_fmt = {l.split()[0]: l.strip().split()[1] for l in list(open('scraping/audio_formats.txt'))}
    if args.low_res:
      video_fmt = {l.split()[0]: l.strip().split()[1] for l in list(open('scraping/video_formats_lowres.txt'))}
    else:
      video_fmt = {l.split()[0]: l.strip().split()[1] for l in list(open('scraping/video_formats.txt'))}

    tasks = [(yid, video_fmt, audio_fmt, args.output_dir) for yid in youtube_ids]
    with poolcontext(processes=7) as pool:
        pbar = tqdm(total=len(tasks))
        for _ in pool.imap_unordered(download_video_audio, tasks):
            update_progress()
    pbar.close()
