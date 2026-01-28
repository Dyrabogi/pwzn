import numpy as np
from scipy.integrate import odeint

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Div
from bokeh.plotting import figure, curdoc

def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdT = -beta / N * I * S
    dIdT = (beta / N * I * S) - (gamma * I)
    dRdT = gamma * I
    return [dSdT, dIdT, dRdT]

def solve_sir(beta, gamma, S0, I0, t_max=200, n_points=1000):
    N = S0 + I0
    R0 = 0.0
    if gamma > 0:
        R0 = beta / gamma
    t = np.linspace(0, t_max, n_points)
    y0 = [S0, I0, 0.0]
    sol = odeint(sir_model, y0, t, args=(beta, gamma, N))
    S, I, R = sol.T
    return t, S, I, R, N, R0

init_beta = 0.3
init_gamma = 0.2
init_S0 = 999
init_I0 = 1
init_tmax = 100

t, S, I, R, N, R0 = solve_sir(init_beta, init_gamma, init_S0, init_I0, init_tmax)


source = ColumnDataSource(data=dict(t=t, S=S, I=I, R=R))


p = figure(
    title=f"SIR model — beta={init_beta}, gamma={init_gamma}",
    x_axis_label='time',
    y_axis_label='population',
    width=700,           
    height=450,
    tools="pan,wheel_zoom,box_zoom,reset"
)

p.line('t', 'S', source=source, line_width=2, legend_label='S(t)', color='blue')
p.line('t', 'I', source=source, line_width=2, legend_label='I(t)', color='red')
p.line('t', 'R', source=source, line_width=2, legend_label='R(t)', color='green')

p.legend.click_policy = "hide"

beta_slider = Slider(start=0.0, end=1.0, value=init_beta, step=0.01, title="β (infection rate)")
gamma_slider = Slider(start=0.01, end=1.0, value=init_gamma, step=0.01, title="γ (recovery rate)")
S0_slider = Slider(start=0, end=5000, value=init_S0, step=1, title="S0 (initial susceptible)")
I0_slider = Slider(start=0, end=5000, value=init_I0, step=1, title="I0 (initial infected)")
tmax_slider = Slider(start=10, end=1000, value=init_tmax, step=1, title="t ")

def update(attr, old, new):
    
    beta = beta_slider.value
    gamma = gamma_slider.value
    S0 = S0_slider.value
    I0 = I0_slider.value
    tmax = tmax_slider.value

    t, S, I, R, N, R0 = solve_sir(beta, gamma, S0, I0, tmax, n_points=1200)

    source.data = dict(t=t, S=S, I=I, R=R)

    p.x_range.start = 0
    p.x_range.end = tmax
    p.y_range.start = 0
    p.y_range.end = max(1.0, N * 1.05)


for w in (beta_slider, gamma_slider, S0_slider, I0_slider, tmax_slider):
    w.on_change('value', update)

controls = column(beta_slider, gamma_slider, S0_slider, I0_slider, tmax_slider, width=300)
layout = row(controls, p)

curdoc().add_root(layout)
curdoc().title = "Interactive SIR model"