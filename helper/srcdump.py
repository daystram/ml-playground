import IPython
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

def dump(filename):
    with open(filename) as f:
        code = f.read()

    formatter = HtmlFormatter()
    return '<style type="text/css">{}</style>{}'.format(
        formatter.get_style_defs('.highlight'),
        highlight(code, PythonLexer(), formatter))
