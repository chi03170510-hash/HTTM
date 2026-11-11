"""Display-only text fitting and collision-free rectangle placement."""

import cv2


def text_width(text, scale=.43):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]


def fit_marked(text, width, scale=.43, marker="..."):
    """Never silently remove text; '!' marks extremely small slots."""
    if text_width(text, scale) <= width:
        return text
    if text_width(marker, scale) > width:
        return "!" if text_width("!", scale) <= width else ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if text_width(text[:middle] + marker, scale) <= width:
            low = middle
        else:
            high = middle - 1
    return text[:low] + marker


def identity_text(identity, pid, width, scale=.38):
    """Priority: accepted SID/Unknown/Ambiguous, then PID. Explicit truncation."""
    full = f"{identity} | {pid}"
    if text_width(full, scale) <= width:
        return full
    if text_width(f"{identity} |  [PID clipped]", scale) <= width:
        return fit_marked(full, width, scale, "... [PID clipped]")
    return fit_marked(identity, width, scale, "... [SID clipped]")


def overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def place_label(desired, size, bounds, occupied):
    """Try nearby free slots; return None instead of overlapping.

    Bounds/rectangles are half-open display pixels. No detection bbox is edited.
    """
    left, top, right, bottom = bounds
    width, height = size
    if width < 1 or height < 1 or width > right - left or height > bottom - top:
        return None
    # Profiled crowded-scene cost: labels outside this image/box region cannot
    # collide with any possible placement inside it, nor supply useful edges.
    occupied = [rect for rect in occupied if overlaps(rect, bounds)]
    clamp_x = lambda x: max(left, min(x, right - width))
    clamp_y = lambda y: max(top, min(y, bottom - height))
    dx, dy = clamp_x(desired[0]), clamp_y(desired[1])
    direct = (dx, dy, dx + width, dy + height)
    if all(not overlaps(direct, old) for old in occupied):
        return direct
    xs = {left, right - width, dx}
    ys = {top, bottom - height, dy}
    for a, b, c, d in occupied:
        xs.update((clamp_x(c + 2), clamp_x(a - width - 2)))
        ys.update((clamp_y(d + 2), clamp_y(b - height - 2)))
    xs = sorted(xs, key=lambda x: (abs(x - dx), x))
    ys = sorted(ys, key=lambda y: (abs(y - dy), y))
    for y in ys:
        for x in xs:
            rect = (x, y, x + width, y + height)
            if all(not overlaps(rect, old) for old in occupied):
                return rect
    return None
