"""
Resample the dissertation figures to the resolution the page actually uses.

The figures were written at the resolution their generators happened to default to, not
at the size they are printed. A plate 8370 pixels wide included at 0.85\\textwidth is
rendered into 12.75cm, which is about 1670 dpi: sixteen times more data than a printer or
a screen can show, carried through every compile. The result was 34MB of PNG, and an
Overleaf build that timed out reading it.

The target is computed per figure rather than applied uniformly, because the figures are
not all included at the same width:

    target pixels = (width fraction) x (text block in inches) x (dpi)

with the text block fixed by the geometry options in main.tex at 15.0cm, or 5.906in. The
width fraction is read from the \\includegraphics call that places the figure, so a plate
included at 0.85\\textwidth and a diagram at 0.7 get different targets and both come out
at the same physical resolution.

Two rules keep this safe to run:

  never upscale   a figure already at or below its target is left untouched, so the tool
                  is idempotent and a second run is a no-op rather than a slow re-encode
                  of everything.

  never crop      resampling is Lanczos on the whole image with the aspect ratio held, so
                  the only thing that changes is how finely it is sampled.

The originals are in git. `git checkout -- Thesis/figures` restores them if a different
resolution is wanted later; rerunning this with another --dpi will not recover detail
already discarded.

Usage:
    python Thesis/tools/downsample_figures.py              # report only
    python Thesis/tools/downsample_figures.py --apply      # resample at 150 dpi
    python Thesis/tools/downsample_figures.py --apply --dpi 200
"""

import glob
import io
import math
import os
import re
import sys

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required: pip install pillow")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FIGDIR = os.path.join(ROOT, "Thesis", "figures")

# main.tex: a4paper, inner=3.5cm, outer=2.5cm -> 21.0 - 6.0 = 15.0cm of text block.
TEXTWIDTH_IN = 15.0 / 2.54
DEFAULT_FRACTION = 0.85          # used when a figure is not placed by any \includegraphics

INC = re.compile(
    r"\\includegraphics\[([^\]]*)\]\{([^}]+)\}")
FRAC = re.compile(r"width\s*=\s*([0-9.]+)\s*\\textwidth")


def placement_fractions():
    """Map figure basename -> the largest width fraction it is included at."""
    out = {}
    pats = [os.path.join(ROOT, "Thesis", d, "*.tex")
            for d in ("chapters", "appendices", "frontmatter")]
    pats.append(os.path.join(ROOT, "Thesis", "*.tex"))
    for pat in pats:
        for path in glob.glob(pat):
            text = io.open(path, encoding="utf-8").read()
            for opts, name in INC.findall(text):
                m = FRAC.search(opts)
                if not m:
                    continue
                key = os.path.basename(name)
                if not key.lower().endswith(".png"):
                    key += ".png"
                out[key] = max(out.get(key, 0.0), float(m.group(1)))
    return out


def human(n):
    return "%.1f MB" % (n / 1e6) if n >= 1e6 else "%.0f kB" % (n / 1e3)


def main():
    apply = "--apply" in sys.argv
    # 300 rather than 150: at 150 the twenty-panel attribution plates render their lead
    # labels as an unreadable smudge, and those labels are the content of the figure.
    # Checked by resampling one plate at each and looking at the result.
    dpi = 300.0
    if "--dpi" in sys.argv:
        dpi = float(sys.argv[sys.argv.index("--dpi") + 1])
    colors = 256
    if "--colors" in sys.argv:
        colors = int(sys.argv[sys.argv.index("--colors") + 1])

    os.chdir(ROOT)
    fracs = placement_fractions()
    files = sorted(glob.glob(os.path.join(FIGDIR, "*.png")))
    if not files:
        raise SystemExit("no PNGs in " + FIGDIR)

    print("resampling target: %.0f dpi over a %.3fin text block\n" % (dpi, TEXTWIDTH_IN))
    print("  %-46s %11s %11s %9s" % ("figure", "before", "after", "saved"))
    print("  " + "-" * 80)

    before_total, after_total, changed, skipped, unplaced = 0, 0, 0, 0, []
    for path in files:
        name = os.path.basename(path)
        size0 = os.path.getsize(path)
        before_total += size0
        frac = fracs.get(name)
        if frac is None:
            unplaced.append(name)
            frac = DEFAULT_FRACTION
        target_w = int(math.ceil(frac * TEXTWIDTH_IN * dpi))

        with Image.open(path) as im:
            w, h = im.size
            needs_resize = w > target_w
            # A matplotlib PNG is 24-bit RGB, but these are flat-colour line plots: the
            # densest carries under 14,000 distinct colours and most of those are text
            # antialiasing. A 256-entry palette reproduces them without visible change --
            # checked on the SHAP beeswarms, whose continuous colourbar is the only place
            # banding could show -- and roughly halves the file again.
            needs_quant = im.mode != "P" and colors > 0
            if not (needs_resize or needs_quant):
                skipped += 1
                after_total += size0
                continue
            if apply:
                im2 = im.convert("RGBA") if im.mode == "P" else im
                if needs_resize:
                    new_h = max(1, int(round(h * target_w / float(w))))
                    im2 = im2.resize((target_w, new_h), Image.LANCZOS)
                if im2.mode == "RGBA":
                    bg = Image.new("RGB", im2.size, (255, 255, 255))
                    bg.paste(im2, mask=im2.split()[-1])
                    im2 = bg
                if needs_quant:
                    im2 = im2.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT)
                im2.save(path, "PNG", optimize=True)

        if apply:
            size1 = os.path.getsize(path)
        else:
            est = size0 * ((target_w / float(w)) ** 2 if needs_resize else 1.0)
            size1 = int(est * (0.42 if needs_quant else 1.0))
        after_total += size1
        changed += 1
        print("  %-46s %11s %11s %8.0f%%"
              % (name[:46], human(size0), human(size1),
                 100.0 * (1 - size1 / float(size0))))

    print("  " + "-" * 80)
    print("  %-46s %11s %11s %8.0f%%"
          % ("total across %d figures" % len(files), human(before_total),
             human(after_total), 100.0 * (1 - after_total / float(before_total))))
    print("\n  resampled %d, already small enough %d" % (changed, skipped))
    if unplaced:
        print("\n  not placed by any includegraphics, assumed %.2f textwidth:" % DEFAULT_FRACTION)
        for n in unplaced:
            print("    %s" % n)
    if not apply:
        print("\n  dry run; after-sizes are estimates. Pass --apply to resample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
