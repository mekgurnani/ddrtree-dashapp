from pathlib import Path
import json
import numpy as np
import os
import matplotlib.pyplot as plt

from collections import defaultdict
import ecg_plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import dash
from dash import dcc, html, Input, Output

# Load Model Information
MODEL_INFO = json.load(open('MODEL_INFO.json'))
i_factors = json.load(open('val_i_factors.json'))
i_factor_ids = [f'{i + 1}' for i in i_factors['indices']]

# Load ecgs and stds
group_medians_3d = np.load('group28_medians_3d.npy')
group_stds_3d = np.load('group28_stds_3d.npy')

decoded_ecgs_array_12L = [group_medians_3d[i] for i in range(group_medians_3d.shape[0])]
decoded_stds_array_12L = [group_stds_3d[i] for i in range(group_stds_3d.shape[0])]

# Code to plot the ecgs
# 12-LEAD ECG LEAD NAMES
ECG_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# COLORMAP TEMPLATES FOR HORIZONTAL AND VERTICAL LEADS
LEAD_COLORS_plotly = {
    'I': 'indigo',
    'II': 'slateblue',
    'III': 'teal',
    'aVR': 'limegreen',
    'aVL': 'midnightblue',
    'aVF': 'mediumturquoise',
    'V1': 'darkslateblue',
    'V2': 'mediumslateblue',
    'V3': 'mediumslateblue',
    'V4': 'mediumslateblue',
    'V5': 'mediumorchid',
    'V6': 'fuchsia'
}
LEAD_COLORS = defaultdict(lambda: 'black', LEAD_COLORS_plotly)


def plot_ecg_plotly(ecg_signal, sampling_rate, lead_names=None, subplot_shape=None, ylim=None, share_ylim=True,
                    title=None, std=None, percentiles=None, figsize=None, show_axes=True, show_legend=False, **kwargs):
    """
    Plots ECG signal(s) in the time domain using Plotly.

    Arguments:
        ecg_signal (ndarray): ECG signal(s) of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal(s).
        lead_names (list): List of lead names. If None, the leads will be named as Lead 1, Lead 2, etc.
        subplot_shape (tuple): Shape of the subplot grid. If None, the shape will be automatically determined.
        ylim (tuple): Y-axis limits of the plot.
        share_ylim (bool): If True, the y-axis limits of the subplots will be shared.
        title (str): Title of the plot.
        std (ndarray): Standard deviation of the ECG signal(s) of shape (num_samples, num_leads).
        percentiles (tuple): Percentiles of the ECG signal(s) of shape (2, num_samples, num_leads).
        figsize (tuple): Figure size in pixels (width, height).
        show_axes (bool): If True, the axes of the plot will be plotted.
        show_legend (bool): If True, the legend will be shown.
        **kwargs: Additional arguments to be passed to the Plotly go.Scatter function.

    Returns:
        fig (plotly.graph_objects.Figure): Figure object.
    """
    # Check ECG signal shape
    if len(ecg_signal.shape) != 2:
        raise ValueError('ECG signal must have shape: (num_samples, num_leads)')

    # Get number of ECG leads and time_index vector
    time_index = np.arange(ecg_signal.shape[0]) / sampling_rate
    num_leads = ecg_signal.shape[1]

    # If share_ylim, find ECG max and min values
    ylim_ = None
    if ylim is not None:
        ylim_ = ylim
    if ylim is None and share_ylim is True:
        ylim_ = (np.min(ecg_signal), np.max(ecg_signal))

    # Check for Lead Names
    if lead_names is not None:
        # Check number of leads
        if len(lead_names) != num_leads:
            raise ValueError('Number of lead names must match the number of leads in the ECG data.')
        lead_colors = LEAD_COLORS_plotly
    else:
        lead_names = [f'Lead {i + 1}' for i in range(num_leads)]  # Lead x
        lead_colors = dict(zip(lead_names, LEAD_COLORS_plotly))

    # Check subplot shape
    if subplot_shape is None:
        subplot_shape = (num_leads, 1)

    subplot_height_cm = 8
    subplot_width_cm = 6.2

    # Calculate relative sizes in Plotly units (normalized)
    row_heights = [subplot_height_cm / (subplot_height_cm * subplot_shape[0])] * subplot_shape[0]
    column_widths = [subplot_width_cm / (subplot_width_cm * subplot_shape[1])] * subplot_shape[1]

    # Create subplots with specified sizes
    fig = make_subplots(
        rows=subplot_shape[0], cols=subplot_shape[1],
        subplot_titles=lead_names,
        row_heights=row_heights,
        column_widths=column_widths,
        vertical_spacing=0.08,
        shared_yaxes=share_ylim,
        x_title="Time (seconds)", y_title="Amplitude (mV)"
    )

    # Plotting
    for i in range(num_leads):
        row = i % subplot_shape[0] + 1
        col = i // subplot_shape[0] + 1

        fig.add_trace(go.Scatter(
            x=time_index,
            y=ecg_signal[:, i],
            mode='lines',
            name=lead_names[i],
            line=dict(color=lead_colors[lead_names[i]]),
            showlegend=show_legend
        ), row=row, col=col)

        fig.update_xaxes(dtick=0.2, row=row, col=col)
        fig.update_yaxes(dtick=0.5, row=row, col=col)

        if std is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([time_index, time_index[::-1]]),
                y=np.concatenate([ecg_signal[:, i] - std[:, i], (ecg_signal[:, i] + std[:, i])[::-1]]),
                fill='toself',
                fillcolor=lead_colors[lead_names[i]],
                line=dict(color='rgba(255,255,255,0)'),
                opacity = 0.2,
                showlegend=False
            ), row=row, col=col)

        if percentiles is not None:
            fig.add_trace(go.Scatter(
                x=time_index,
                y=percentiles[0][:, i],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=time_index,
                y=percentiles[1][:, i],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=lead_colors[lead_names[i]],
                opacity=0.2,
                showlegend=False
            ), row=row, col=col)

    # Update layout
    fig.update_layout(
        title=title,
        width=subplot_shape[1] * subplot_width_cm * 40,
        height=subplot_shape[0] * subplot_height_cm * 40,
    )

    if ylim_ is not None:
        fig.update_yaxes(range=ylim_)

    if not show_axes:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

    return fig

# Make the dash app

broadqrs_ddrtree = pd.read_csv("tree_proj_full_branches_plotly_relevant.csv")

# Map branch coordinates to colors
color_map = {
    1: 'firebrick',
    2: 'gold',
    3: 'forestgreen',
    4: 'lightseagreen',
    5: 'royalblue',
    6: 'orchid'
}

branch_map = {
    1: 'Higher risk LBBB (Phenogroup 1)',
    2: 'Higher risk LBBB/NSIVCD (Phenogroup 2)',
    3: 'Higher risk IVCD (Phenogroup 3)',
    4: 'Average branch RBBB (Phenogroup 4)',
    5: 'Lower risk RBBB (Phenogroup 5)',
    6: 'Higher risk RBBB (Phenogroup 6)'
}

color_map1 = {
    'firebrick': (178, 34, 34),
    'gold': (255, 215, 0),
    'forestgreen': (34, 139, 34),
    'lightseagreen': (32, 178, 170),
    'royalblue': (65, 105, 225),
    'orchid': (218, 112, 214)
}

tips_type_map = {
    "pheno1_left1": 0,
    "pheno1_left2": 1,
    "pheno1_right1": 2,
    "pheno1_right2": 3,
    "pheno2_1": 4,
    "pheno2_2": 5,
    "pheno2_31": 6,
    "pheno2_32": 7,
    "pheno2_4": 8,
    "pheno2_5": 9,
    "pheno3_1": 10,
    "pheno3_2": 11,
    "pheno3_3": 12,
    "pheno3_41": 13,
    "pheno3_42": 14,
    "pheno4_1": 15,
    "pheno4_2": 16,
    "pheno4_3": 17,
    "pheno5_1": 18,
    "pheno5_2": 19,
    "pheno5_31": 20,
    "pheno5_32": 21,
    "pheno5_4": 22,
    "pheno5_5": 23,
    "pheno5_6": 24,
    "pheno6_1": 25,
    "pheno6_2": 26,
    "pheno6_3": 27
}

broadqrs_ddrtree['phenogroup'] = broadqrs_ddrtree['merged_branchcoords'].map(branch_map)
broadqrs_ddrtree['tips_type_mapped'] = broadqrs_ddrtree['tips_type'].map(tips_type_map)
broadqrs_ddrtree['tips_type_mapped'] = broadqrs_ddrtree['tips_type_mapped'].astype('Int64')

# Add a new column 'color' with mapped colors
broadqrs_ddrtree['color'] = broadqrs_ddrtree['merged_branchcoords'].map(color_map).fillna('gray')

# Create initial Plotly figure
fig = go.Figure()

# Add the scattergl trace
fig.add_trace(go.Scattergl(
    x=broadqrs_ddrtree['Z1'],
    y=broadqrs_ddrtree['Z2'],
    mode='markers',
    marker=dict(
        size=7,
        color=broadqrs_ddrtree['color'],
        opacity=0.7,
        line=dict(width=1, color='black')
    ),
    name='Scatter Points',
    hoverinfo='x+y+text',
    text=broadqrs_ddrtree['phenogroup']
))

# Customize the layout
fig.update_layout(
    title='Broad QRS DDRTree',
    xaxis_title='Dimension 1',
    yaxis_title='Dimension 2',
    width=700,
    height=700,
    font=dict(
        size=15
    )
)

# Create Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1(children='Visualising ECGs from the broad QRS DDRTree', style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure=fig
        ),
        dcc.Graph(
            id='hover-data-plot',
            style={'margin-top': '1px'}  # Adjust the margin-top value as needed
        )
    ], style={'display': 'flex'}),
])

@app.callback(
    Output('hover-data-plot', 'figure'),
    [Input('scatter-plot', 'hoverData')]
)
def update_hover_plot(hoverData):
    if hoverData is None:
        # If there is no hover data, return a blank plot with just the text annotation
        return {
            'data': [],
            'layout': {
                'annotations': [
                    {
                        'x': 3,
                        'y': 1.5,
                        'text': "Hover over a point on the tree to see the reconstructed ECG.",
                        'showarrow': False,
                        'font': {'size': 13, 'color': 'black'},
                    }
                ],
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'width': 400,
                'height': 600,
            }
        }
    
    # If hover data is available, proceed to retrieve and plot the ECG data
    point_index = hoverData['points'][0]['pointIndex']
    tips_type = broadqrs_ddrtree.iloc[point_index]['tips_type_mapped']
    
    # Retrieve the ECG data for the hovered point
    ecg_data = decoded_ecgs_array_12L[tips_type]
    lead_names = ['I', 'aVR', 'V1', 'V4',
                'II', 'aVL', 'V2', 'V5',
                'III', 'aVF', 'V3', 'V6']

    # Generate the ECG plot using viz.plot_ecg
    fig_plotly = plot_ecg_plotly(ecg_data, 400, lead_names=lead_names, subplot_shape=(3, 4), ylim=(-2, 2), subplots=True, figsize=(1000, 1000), std = decoded_stds_array_12L[tips_type])

    # Customize the layout to show the axes and title based on broadqrs_ddrtree color
    phenogroup = broadqrs_ddrtree['phenogroup'].iloc[point_index]
    point_color_name = hoverData['points'][0]['marker.color']
    rgb_color = color_map1.get(point_color_name)
    plotly_color = f'rgb{rgb_color}'
    fig_plotly['layout']['title'].update(text=f"Reconstructed ECG - {phenogroup}", font=dict(color=plotly_color, size=15))
    
    return fig_plotly

if __name__ == "__main__":
    app.run_server(debug=True)