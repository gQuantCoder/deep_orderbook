# visu.py

import os
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import clear_output, display


def _widget_env() -> bool:
    """True when the notebook runtime supports ipywidgets (JupyterLab/classic).
    VS Code and Cursor have VSCODE_* env vars and don't render FigureWidget."""
    return not any(k.startswith("VSCODE_") for k in os.environ)


class Visualizer:
    """Live-updating order-book visualiser.

    JupyterLab  → FigureWidget + batch_update(): diff-only, smooth 10+ FPS.
    Cursor/Code → go.Figure + clear_output(): throttled full redraw.

    Run in JupyterLab for the best experience:  uv run jupyter lab
    """

    def __init__(self, refresh_interval: float = 0.1) -> None:
        self._widgets = _widget_env()
        self.fig: go.FigureWidget | go.Figure = self._create_figure()
        self._initialize_traces()
        display(self.fig)
        self.losses: list[float] = []
        self.test_losses: list[float] = []
        self._max_points = 605
        self._loss_max_points = 128
        # fallback throttle used only in non-widget envs
        self._refresh_interval = refresh_interval
        self._last_refresh: float = 0.0

    def _refresh(self) -> None:
        """No-op in widget envs (batch_update pushes diffs automatically).
        In Cursor/Code: throttled clear+redraw."""
        if self._widgets:
            return
        now = time.monotonic()
        if now - self._last_refresh < self._refresh_interval:
            return
        self._last_refresh = now
        clear_output(wait=True)
        display(self.fig)

    def _create_figure(self) -> go.FigureWidget | go.Figure:
        """Creates and returns a Plotly FigureWidget with subplots."""
        fig = make_subplots(
            rows=6,  # Added one more row for PnL
            cols=1,
            subplot_titles=(
                "Bid and Ask Price Levels",
                "Books",
                "Level Proximity",
                "Prediction",
                "Training Loss vs Test Loss",
                "Omniscient PnL vs Prediction PnL",
            ),
            vertical_spacing=0.05,
            row_heights=[0.16] * 6,  # Adjusted heights for 6 subplots
        )

        fig.update_layout(
            height=1000,  # Increased height to accommodate new subplot
            width=1200,
            margin=dict(t=50, b=50, l=50, r=50),
            showlegend=True,
            legend=dict(
                orientation="h",  # horizontal legend
                yanchor="bottom",
                y=1.02,  # position above the plot
                xanchor="center",
                x=0.5,  # center horizontally
            ),
            # Configure all x-axes
            xaxis=dict(
                domain=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey',
                zeroline=False,
                fixedrange=True,
                dtick=100,
                showticklabels=True,
                range=[0, 1000],
            ),
            xaxis2=dict(
                domain=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey',
                zeroline=False,
                fixedrange=True,
                # showticklabels=False,  # Hide labels for other subplots
            ),
            xaxis3=dict(
                domain=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey',
                zeroline=False,
                fixedrange=True,
                # showticklabels=False,
            ),
            xaxis4=dict(
                domain=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey',
                zeroline=False,
                fixedrange=True,
                showticklabels=False,
            ),
            xaxis5=dict(
                domain=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey',
                zeroline=False,
                fixedrange=True,
                # showticklabels=False,
            ),
            xaxis6=dict(
                domain=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey',
                zeroline=False,
                fixedrange=True,
                # showticklabels=False,
            ),
            # Configure y-axes
            yaxis=dict(
                title=dict(text="Price", font=dict(color="black")),
                tickfont=dict(color="black"),
                domain=[0.875, 1.0],
                tickformat=".2f",
                fixedrange=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey',
                zeroline=False,
                autorange=True,
            ),
            # Add a secondary y-axis for test loss
            yaxis5=dict(
                title=dict(text="Training Loss", font=dict(color="blue")),
                tickfont=dict(color="blue"),
                domain=[0.175, 0.3],
            ),
            yaxis6=dict(
                title=dict(text="Test Loss", font=dict(color="red")),
                tickfont=dict(color="red"),
                anchor="x5",
                overlaying="y5",
                side="right",
            ),
            yaxis7=dict(
                title=dict(text="Omniscient PnL", font=dict(color="green")),
                tickfont=dict(color="green"),
                domain=[0.0, 0.16],
                tickformat="02d",
                fixedrange=True,
            ),
            yaxis8=dict(
                title=dict(text="Prediction PnL", font=dict(color="red")),
                tickfont=dict(color="red"),
                anchor="x6",
                overlaying="y7",
                side="right",
                tickformat="02d",
                fixedrange=True,
            ),
        )

        return go.FigureWidget(fig) if self._widgets else go.Figure(fig)

    def _initialize_traces(self) -> None:
        """Initializes and adds traces to the figure."""
        # Line traces for Bid and Ask Price Levels
        self.bid_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="green"),
            showlegend=False,
        )
        self.ask_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red"),
            showlegend=False,
        )
        # Position entry markers (triangles up)
        self.entry_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="blue",
            ),
            name="Position Entry",
            showlegend=True,
        )
        # Position exit markers (triangles down)
        self.exit_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="red",
            ),
            name="Position Exit",
            showlegend=True,
        )

        # Heatmap for Books
        self.im_trace = go.Heatmap(
            z=np.zeros((10, 10)),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            showscale=False,
        )

        # Add scatter traces for 2nd and 3rd feature dimensions
        self.feature2_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                # size=12,
                color="rgba(0, 0, 0, 0.8)",
            ),
            name="Feature 2",
            showlegend=True,
        )

        self.feature3_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                # size=12,
                color="rgba(0, 0, 0, 0.8)",
            ),
            name="Feature 3",
            showlegend=True,
        )

        # Heatmap for Level Proximity
        self.t2l_trace = go.Heatmap(
            z=np.zeros((10, 10)),
            colorscale="Turbo",
            zmin=0,
            zmax=1,
            showscale=False,
        )

        # Heatmap for Prediction
        self.pred_trace = go.Heatmap(
            z=np.zeros((10, 10)),
            colorscale="Turbo",
            zmin=0,
            zmax=1,
            showscale=False,
        )

        # Line trace for Training Loss (left y-axis)
        self.loss_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="blue", width=2),
            name="Training Loss",
            showlegend=True,
            yaxis="y5",
        )

        # Line trace for Test Loss (right y-axis)
        self.test_loss_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=2),
            name="Test Loss",
            showlegend=True,
            yaxis="y6",
        )

        # Line trace for Omniscient PnL (left y-axis)
        self.gt_pnl_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="green", width=2),
            name="Omniscient PnL",
            showlegend=True,
            yaxis="y7",
        )

        # Line trace for Prediction PnL (right y-axis)
        self.pred_pnl_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=2),
            name="Prediction PnL",
            showlegend=True,
            yaxis="y8",
        )

        # Line traces for up/down proximity signals
        self.up_proximity_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=2),
            name="Up Proximity",
            showlegend=True,
            yaxis="y4",  # Same y-axis as prediction heatmap
        )
        self.down_proximity_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="green", width=2),
            name="Down Proximity",
            showlegend=True,
            yaxis="y4",  # Same y-axis as prediction heatmap
        )

        # Ground truth position entry markers (triangles up)
        self.gt_entry_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="green",
            ),
            name="Omniscient Entry",
            showlegend=True,
        )
        # Ground truth position exit markers (triangles down)
        self.gt_exit_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="darkgreen",
            ),
            name="Omniscient Exit",
            showlegend=True,
        )
        # Predicted position entry markers (triangles up)
        self.pred_entry_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="red",
            ),
            name="Prediction Entry",
            showlegend=True,
        )
        # Predicted position exit markers (triangles down)
        self.pred_exit_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="darkred",
            ),
            name="Prediction Exit",
            showlegend=True,
        )

        # Add all traces to the figure widget
        self.fig.add_trace(self.bid_trace, row=1, col=1)
        self.fig.add_trace(self.ask_trace, row=1, col=1)
        self.fig.add_trace(self.gt_entry_trace, row=1, col=1)
        self.fig.add_trace(self.gt_exit_trace, row=1, col=1)
        self.fig.add_trace(self.pred_entry_trace, row=1, col=1)
        self.fig.add_trace(self.pred_exit_trace, row=1, col=1)
        self.fig.add_trace(self.im_trace, row=2, col=1)
        self.fig.add_trace(self.feature2_trace, row=2, col=1)
        self.fig.add_trace(self.feature3_trace, row=2, col=1)
        self.fig.add_trace(self.t2l_trace, row=3, col=1)
        self.fig.add_trace(self.pred_trace, row=4, col=1)
        self.fig.add_trace(self.up_proximity_trace, row=4, col=1)
        self.fig.add_trace(self.down_proximity_trace, row=4, col=1)
        self.fig.add_trace(self.loss_trace, row=5, col=1)
        self.fig.add_trace(self.test_loss_trace, row=5, col=1)
        self.fig.add_trace(self.gt_pnl_trace, row=6, col=1)
        self.fig.add_trace(self.pred_pnl_trace, row=6, col=1)

    def update(
        self,
        books_z_data: np.ndarray | None,
        level_reach_z_data: np.ndarray | None,
        bidask: np.ndarray | None,
        pred_t2l: np.ndarray | None = None,
        gt_pnl: np.ndarray | None = None,
        pred_pnl: np.ndarray | None = None,
        positions: np.ndarray | None = None,  # Ground truth positions
        pred_positions: np.ndarray | None = None,  # Predicted positions
        up_proximity: np.ndarray | None = None,  # Ground truth up proximity
        down_proximity: np.ndarray | None = None,  # Ground truth down proximity
        pred_up_proximity: np.ndarray | None = None,  # Predicted up proximity
        pred_down_proximity: np.ndarray | None = None,  # Predicted down proximity
    ) -> None:
        """Updates the figure with new data."""
        try:
            with self.fig.batch_update():
                # Transform and clip all image data
                books_z_data, level_reach_display, bidask, feature_points = (
                    self.for_image_display(
                        books_z_data,
                        level_reach_z_data,
                        bidask,
                        max_points=self._max_points,
                    )
                )
                # Transform prediction data in the same way
                _, pred_t2l_display, _, _ = (
                    self.for_image_display(
                        None, pred_t2l, None, max_points=self._max_points
                    )
                    if pred_t2l is not None
                    else (None, None, None, None)
                )

                # Update bid and ask price traces with limited history
                if bidask is not None:
                    times = np.arange(min(bidask.shape[0], self._max_points))
                    bid_data = (
                        bidask[-self._max_points :, 0]
                        if bidask.shape[0] > self._max_points
                        else bidask[:, 0]
                    )
                    ask_data = (
                        bidask[-self._max_points :, 1]
                        if bidask.shape[0] > self._max_points
                        else bidask[:, 1]
                    )

                    # Update x-axis range to match the data for all subplots
                    x_range = [times[0], times[-1]]
                    for i in range(1, 5):  # Update all 6 x-axes
                        self.fig.layout[f'xaxis{i}'].range = x_range

                    self.fig.data[0].x = times
                    self.fig.data[0].y = bid_data
                    self.fig.data[1].x = times
                    self.fig.data[1].y = ask_data

                    # Update ground truth position markers
                    if positions is not None:
                        pos_data = (
                            positions[-self._max_points :]
                            if len(positions) > self._max_points
                            else positions
                        )
                        entry_indices = (
                            np.where((pos_data[1:] == 1) & (pos_data[:-1] == 0))[0] + 1
                        )
                        exit_indices = (
                            np.where((pos_data[1:] == 0) & (pos_data[:-1] == 1))[0] + 1
                        )

                        # Update ground truth entry markers - place at ask price for buys
                        self.fig.data[2].x = times[entry_indices]
                        self.fig.data[2].y = ask_data[entry_indices]

                        # Update ground truth exit markers - place at bid price for sells
                        self.fig.data[3].x = times[exit_indices]
                        self.fig.data[3].y = bid_data[exit_indices]

                    # Update predicted position markers
                    if pred_positions is not None:
                        pred_pos_data = (
                            pred_positions[-self._max_points :]
                            if len(pred_positions) > self._max_points
                            else pred_positions
                        )
                        pred_entry_indices = (
                            np.where(
                                (pred_pos_data[1:] == 1) & (pred_pos_data[:-1] == 0)
                            )[0]
                            + 1
                        )
                        pred_exit_indices = (
                            np.where(
                                (pred_pos_data[1:] == 0) & (pred_pos_data[:-1] == 1)
                            )[0]
                            + 1
                        )

                        # Update predicted entry markers - place at ask price for buys
                        self.fig.data[4].x = times[pred_entry_indices]
                        self.fig.data[4].y = ask_data[pred_entry_indices]

                        # Update predicted exit markers - place at bid price for sells
                        self.fig.data[5].x = times[pred_exit_indices]
                        self.fig.data[5].y = bid_data[pred_exit_indices]

                # Update heatmaps
                if books_z_data is not None:
                    self.fig.data[6].z = books_z_data

                    # Update feature markers if available
                    if feature_points is not None:
                        feature2_points, feature3_points = feature_points

                        # Update feature 2 markers (up triangles)
                        self.fig.data[7].x = feature2_points[0]  # time dimension
                        self.fig.data[7].y = feature2_points[
                            1
                        ]  # price level dimension

                        # Update feature 3 markers (down triangles)
                        self.fig.data[8].x = feature3_points[0]  # time dimension
                        self.fig.data[8].y = feature3_points[
                            1
                        ]  # price level dimension

                if level_reach_display is not None:
                    self.fig.data[9].z = level_reach_display
                if pred_t2l_display is not None:
                    self.fig.data[10].z = pred_t2l_display

                    # Update ground truth proximity traces if available
                    if up_proximity is not None and down_proximity is not None:
                        times = np.arange(len(up_proximity))[: self._max_points]
                        up_prox_data = up_proximity[-self._max_points :]
                        down_prox_data = down_proximity[-self._max_points :]
                        self.fig.data[11].x = times
                        self.fig.data[11].y = np.clip(up_prox_data, 0, 1) * (
                            pred_t2l_display.shape[0] - 1
                        )
                        self.fig.data[12].x = times
                        self.fig.data[12].y = np.clip(down_prox_data, 0, 1) * (
                            pred_t2l_display.shape[0] - 1
                        )

                    # Update predicted proximity traces if available
                    if (
                        pred_up_proximity is not None
                        and pred_down_proximity is not None
                    ):
                        times = np.arange(len(pred_up_proximity))[: self._max_points]
                        pred_up_prox_data = pred_up_proximity[-self._max_points :]
                        pred_down_prox_data = pred_down_proximity[-self._max_points :]
                        self.fig.data[11].x = times
                        self.fig.data[11].y = np.clip(
                            pred_up_prox_data, 0, 1
                        ) * (pred_t2l_display.shape[0] - 1)
                        self.fig.data[12].x = times
                        self.fig.data[12].y = np.clip(
                            pred_down_prox_data, 0, 1
                        ) * (pred_t2l_display.shape[0] - 1)

                # Update loss traces
                if self.losses:
                    loss_times = np.arange(len(self.losses))[-self._loss_max_points :]
                    self.fig.data[13].x = loss_times
                    self.fig.data[13].y = self.losses[-self._loss_max_points :]

                if self.test_losses:
                    test_loss_times = np.arange(len(self.test_losses))[
                        -self._loss_max_points :
                    ]
                    self.fig.data[14].x = test_loss_times
                    self.fig.data[14].y = self.test_losses[
                        -self._loss_max_points :
                    ]

                # Update PnL traces
                if gt_pnl is not None:
                    pnl_times = np.arange(len(gt_pnl))
                    self.fig.data[15].x = pnl_times
                    self.fig.data[15].y = gt_pnl[-self._max_points :]
                    self.fig.data[15].yaxis = "y7"

                if pred_pnl is not None:
                    pred_pnl_times = np.arange(len(pred_pnl))
                    self.fig.data[16].x = pred_pnl_times
                    self.fig.data[16].y = pred_pnl[-self._max_points :]
                    self.fig.data[16].yaxis = "y8"

            self._refresh()

        except Exception as e:
            print(f"Error updating plot: {e}")

    def add_loss(self, train_loss: float | None, test_loss: float) -> None:
        """Adds loss values to the loss history.

        Args:
            train_loss: Training loss value, can be None if only test loss is available
            test_loss: Test loss value
        """
        if train_loss is not None:
            self.losses.append(
                float(train_loss)
            )  # Convert to float to ensure no reference holding
            # Keep only recent history
            if len(self.losses) > self._max_points:
                self.losses = self.losses[-self._max_points :]

        self.test_losses.append(float(test_loss))
        # Keep only recent history
        if len(self.test_losses) > self._max_points:
            self.test_losses = self.test_losses[-self._max_points :]

    @staticmethod
    def for_image_display(
        books_array: np.ndarray | None = None,
        t2l_array: np.ndarray | None = None,
        prices_array: np.ndarray | None = None,
        max_points: int = 605,  # Default to _max_points value
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
    ]:
        im_data: np.ndarray = np.ndarray(shape=(0, 0))
        t2l_data: np.ndarray = np.ndarray(shape=(0, 0))
        feature_points = None

        if books_array is not None:
            # Take last max_points if array is longer
            if books_array.shape[0] > max_points:
                books_array = books_array[-max_points:]
            im_data = books_array.copy()
            im_data[:, :, 0] *= 0.5
            im_data = im_data.mean(axis=2).T
            im_data = np.clip(im_data, -1, 1)

            # Extract non-zero positions for features 2 and 3
            feature2_nonzero = np.nonzero(books_array[:, :, 1] != 0)
            feature3_nonzero = np.nonzero(books_array[:, :, 2] != 0)
            feature_points = (
                (feature2_nonzero[0], feature2_nonzero[1]),
                (feature3_nonzero[0], feature3_nonzero[1]),
            )

        if t2l_array is not None:
            # Take last max_points if array is longer
            if t2l_array.shape[0] > max_points:
                t2l_array = t2l_array[-max_points:]
            t2l_data = t2l_array[:, :, 0].T
            t2l_data = np.clip(t2l_data, -1, 1)

        return im_data, t2l_data, prices_array, feature_points


if __name__ == '__main__':
    vis = Visualizer()
    rng = np.random.default_rng(456456)
    data_dict = {
        "books_z_data": rng.random((10, 10, 3)),
        "level_reach_z_data": rng.random((10, 10, 1)),
        "bidask": rng.random((10, 2)),
    }
    vis.update(**data_dict)
