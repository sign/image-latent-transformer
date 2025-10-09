FROM continuumio/miniconda3:latest

# Setup package manager
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.sources && \
    apt-get update -q

### Install PyCairo dependencies
#RUN apt-get install -y libcairo2-dev
#
### Install PyGObject dependencies
#RUN apt-get install -y gir1.2-gtk-4.0

RUN apt-get install -y gobject-introspection libgirepository-1.0-1 \
    libcairo2 libcairo-gobject2 libpango-1.0-0 libpangocairo-1.0-0 \
    gir1.2-pango-1.0 gir1.2-cairo-1.0 gir1.2-gtk-4.0

RUN conda install python=3.12 -y

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda tos accept --override-channels --channel conda-forge

RUN conda install -c conda-forge pycairo pygobject manimpango -y

RUN set -eux; cat > test_pango.py <<'PY'
import gi
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
gi.require_foreign("cairo")
from gi.repository import Pango, PangoCairo, cairo
import cairo

surface = cairo.ImageSurface(cairo.Format.ARGB32, 1, 1)
ctx = cairo.Context(surface)
layout = PangoCairo.create_layout(ctx)
layout.set_text("Hello world", -1)
layout.set_font_description(Pango.font_description_from_string("sans 12"))
print("✅ Successfully created layout and set text.")
PY

CMD ["python", "test_pango.py"]

# docker build -t renderer-welt -f renderer.Dockerfile .
# docker run -it --rm renderer-welt