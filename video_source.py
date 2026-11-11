"""Validation for supported single-video sources, shared by the API and worker."""
import re
from urllib.parse import urlsplit, parse_qs, urlencode


def normalize_video_url(value):
    match = re.search(r'https?://[^\s<>\[\]"\u3000]+', value.strip())
    if not match:
        raise ValueError('请输入完整的视频链接，也可以粘贴包含链接的分享文字')
    url = match.group(0).rstrip(')）。，,;；').replace('\\&', '&')
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.port not in (None, 80, 443):
        raise ValueError('视频链接格式无效')
    host = (parsed.hostname or '').lower()
    if host in ('bilibili.com', 'www.bilibili.com', 'm.bilibili.com'):
        video = re.fullmatch(r'/video/(BV[A-Za-z0-9]{10}|av\d+)/?', parsed.path, flags=re.I)
        if not video:
            raise ValueError('请使用 B 站单个视频链接，例如 https://www.bilibili.com/video/BV…/')
        video_id = video.group(1)
        video_id = ('BV' if video_id[:2].lower() == 'bv' else 'av') + video_id[2:]
        result = 'https://www.bilibili.com/video/' + video_id + '/'
        page = parse_qs(parsed.query).get('p')
        if page:
            if len(page) != 1 or not page[0].isdigit() or not 1 <= int(page[0]) <= 10000:
                raise ValueError('B 站分 P 编号必须是正整数')
            result += '?' + urlencode({'p':int(page[0])})
        return result
    if host == 'b23.tv' and re.fullmatch(r'/[A-Za-z0-9_-]+/?', parsed.path):
        return 'https://b23.tv' + parsed.path
    if parsed.scheme == 'https' and (host == 'douyinvod.com' or host.endswith('.douyinvod.com')) and parsed.path not in ('', '/'):
        return url  # Signed CDN URLs require the complete query string.
    if host in ('douyin.com', 'www.douyin.com'):
        video = re.fullmatch(r'/video/(\d+)/?', parsed.path)
        if not video:
            raise ValueError('请使用抖音单个视频链接，例如 https://www.douyin.com/video/视频编号')
        return 'https://www.douyin.com/video/' + video.group(1)
    if host == 'v.douyin.com' and re.fullmatch(r'/[A-Za-z0-9_-]+/?', parsed.path):
        return 'https://v.douyin.com' + parsed.path
    raise ValueError('当前支持抖音和 Bilibili 视频页面、短链接，以及 douyinvod.com 媒体直链')
