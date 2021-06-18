import plotly.express as px

def scatter_(df, x, y, c):
    if c == 'None':
        fig = px.scatter(df, x=x, y=y)
    else:
        fig = px.scatter(df, x=x, y=y, color=c)
    return fig

def histogram_(df, x, c, bins):
    if c == 'None':
        fig = px.histogram(df, x=x, nbins=bins)
    else:
        fig = px.histogram(df, x=x, color=c, nbins=bins)
    return fig

def box_(df, x, c):
    fig = px.box(df, x=c, y=x, color = c, notched=True)
    return fig
