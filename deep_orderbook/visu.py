# visu.py

import os
import time

import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output, display
from plotly.subplots import make_subplots


def _widget_env() -> bool:
    """True when the notebook runtime supports ipywidgets (JupyterLab/classic).
    VS Code and Cursor have VSCODE_* env vars and don't render FigureWidget."""
    return not any(k.startswith("VSCODE_") for k in os.environ)


# Module-level label set by the training loop; read by Visualizer.update().
_window_label: str = ""


def set_window_label(label: str) -> None:
    global _window_label
    _window_label = label


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
            rows=6,
            cols=1,
            subplot_titles=(
                "Bid and Ask Price Levels",
                "Books",
                "Level Proximity",
                "Prediction",
                "Loss",
                "PnL",
            ),
            vertical_spacing=0.04,
            row_heights=[0.22, 0.18, 0.18, 0.18, 0.12, 0.12],
        )
        fig.update_layout(
            height=1100,
            width=1200,
            margin=dict(t=50, b=30, l=50, r=50),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        # Uniform x-axis style; row 1 starts fixed at [0, 1000] and is updated each frame.
        fig.update_xaxes(showgrid=True, gridcolor="lightgrey", zeroline=False, fixedrange=True)
        fig.update_xaxes(range=[0, 1000], row=1, col=1)
        # Window label annotation — top-right corner, updated each frame.
        fig.add_annotation(
            text="",
            xref="paper", yref="paper",
            x=1.0, y=1.0,
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font={"size": 11, "color": "dimgray"},
        )
        # Y-axis labels
        fig.update_yaxes(title_text="Price", tickformat=".2f", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=5, col=1)
        fig.update_yaxes(title_text="PnL", row=6, col=1)

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

        # Line trace for Training Loss
        self.loss_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="blue", width=2),
            name="Training Loss",
            showlegend=True,
        )

        # Line trace for Test Loss
        self.test_loss_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=2),
            name="Test Loss",
            showlegend=True,
        )

        # Line trace for Omniscient PnL
        self.gt_pnl_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="green", width=2),
            name="Omniscient PnL",
            showlegend=True,
        )

        # Line trace for Prediction PnL
        self.pred_pnl_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=2),
            name="Prediction PnL",
            showlegend=True,
        )

        # Line traces for up/down proximity signals
        self.up_proximity_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=2),
            name="Up Proximity",
            showlegend=True,
        )
        self.down_proximity_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="green", width=2),
            name="Down Proximity",
            showlegend=True,
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
        # Predicted position entry markers: short → triangle-down (selling)
        self.pred_entry_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="red",
            ),
            name="Prediction Entry",
            showlegend=True,
        )
        # Predicted position exit markers: short → triangle-up (covering)
        self.pred_exit_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
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

        # Rebind named attributes to the live figure trace objects.
        # fig.add_trace() copies data into the figure's internal store;
        # only fig.data[i] is the live reference that batch_update() can push.
        (
            self.bid_trace,
            self.ask_trace,
            self.gt_entry_trace,
            self.gt_exit_trace,
            self.pred_entry_trace,
            self.pred_exit_trace,
            self.im_trace,
            self.feature2_trace,
            self.feature3_trace,
            self.t2l_trace,
            self.pred_trace,
            self.up_proximity_trace,
            self.down_proximity_trace,
            self.loss_trace,
            self.test_loss_trace,
            self.gt_pnl_trace,
            self.pred_pnl_trace,
        ) = self.fig.data

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
                if _window_label:
                    self.fig.layout.annotations[-1].text = _window_label
                # Transform and clip all image data
                books_z_data, level_reach_display, bidask, feature_points = self.for_image_display(
                    books_z_data,
                    level_reach_z_data,
                    bidask,
                    max_points=self._max_points,
                )
                # Transform prediction data in the same way
                _, pred_t2l_display, _, _ = (
                    self.for_image_display(None, pred_t2l, None, max_points=self._max_points)
                    if pred_t2l is not None
                    else (None, None, None, None)
                )

                # Use robust shared scaling for true/pred proximity maps so the
                # live notebook view matches the saved experiment dashboards more
                # closely and does not collapse into black/red saturation.
                proximity_vmax = None
                if level_reach_display is not None or pred_t2l_display is not None:
                    candidates = []
                    if level_reach_display is not None and level_reach_display.size:
                        candidates.append(float(np.percentile(level_reach_display, 99.5)))
                    if pred_t2l_display is not None and pred_t2l_display.size:
                        candidates.append(float(np.percentile(pred_t2l_display, 99.5)))
                    proximity_vmax = max(candidates + [1e-3])

                # Update bid and ask price traces with limited history
                if bidask is not None:
                    times = np.arange(min(bidask.shape[0], self._max_points))
                    bid_data = bidask[-self._max_points :, 0] if bidask.shape[0] > self._max_points else bidask[:, 0]
                    ask_data = bidask[-self._max_points :, 1] if bidask.shape[0] > self._max_points else bidask[:, 1]

                    # Update x-axis range for the data subplots (rows 1–4).
                    # Row 1 uses "xaxis" (no suffix); rows 2–4 use "xaxis2"–"xaxis4".
                    x_range = [times[0], times[-1]]
                    for ax_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
                        self.fig.layout[ax_name].range = x_range

                    self.bid_trace.x = times
                    self.bid_trace.y = bid_data
                    self.ask_trace.x = times
                    self.ask_trace.y = ask_data

                    # Update ground truth position markers
                    if positions is not None:
                        pos_data = positions[-self._max_points :] if len(positions) > self._max_points else positions
                        # Works for both long (+1) and short (-1): any flat→in-pos transition
                        entry_indices = np.where((pos_data[:-1] == 0) & (pos_data[1:] != 0))[0] + 1
                        exit_indices = np.where((pos_data[:-1] != 0) & (pos_data[1:] == 0))[0] + 1

                        self.gt_entry_trace.x = times[entry_indices]
                        self.gt_entry_trace.y = ask_data[entry_indices]
                        self.gt_exit_trace.x = times[exit_indices]
                        self.gt_exit_trace.y = bid_data[exit_indices]

                    # Update predicted position markers
                    if pred_positions is not None:
                        pred_pos_data = (
                            pred_positions[-self._max_points :] if len(pred_positions) > self._max_points else pred_positions
                        )
                        # Works for both long (+1) and short (-1): any flat→in-pos transition
                        pred_entry_indices = np.where((pred_pos_data[:-1] == 0) & (pred_pos_data[1:] != 0))[0] + 1
                        pred_exit_indices = np.where((pred_pos_data[:-1] != 0) & (pred_pos_data[1:] == 0))[0] + 1

                        # Short entry: sell at bid; short exit: buy back at ask
                        self.pred_entry_trace.x = times[pred_entry_indices]
                        self.pred_entry_trace.y = bid_data[pred_entry_indices]
                        self.pred_exit_trace.x = times[pred_exit_indices]
                        self.pred_exit_trace.y = ask_data[pred_exit_indices]

                # Update heatmaps
                if books_z_data is not None:
                    self.im_trace.z = books_z_data

                    # Update feature markers if available
                    if feature_points is not None:
                        feature2_points, feature3_points = feature_points

                        self.feature2_trace.x = feature2_points[0]
                        self.feature2_trace.y = feature2_points[1]
                        self.feature3_trace.x = feature3_points[0]
                        self.feature3_trace.y = feature3_points[1]

                if level_reach_display is not None:
                    self.t2l_trace.z = level_reach_display
                    if proximity_vmax is not None:
                        self.t2l_trace.zmin = 0
                        self.t2l_trace.zmax = proximity_vmax
                if pred_t2l_display is not None:
                    self.pred_trace.z = pred_t2l_display
                    if proximity_vmax is not None:
                        self.pred_trace.zmin = 0
                        self.pred_trace.zmax = proximity_vmax

                    # Update ground truth proximity traces if available
                    if up_proximity is not None and down_proximity is not None:
                        times = np.arange(len(up_proximity))[: self._max_points]
                        up_prox_data = up_proximity[-self._max_points :]
                        down_prox_data = down_proximity[-self._max_points :]
                        self.up_proximity_trace.x = times
                        self.up_proximity_trace.y = np.clip(up_prox_data, 0, 1) * (pred_t2l_display.shape[0] - 1)
                        self.down_proximity_trace.x = times
                        self.down_proximity_trace.y = np.clip(down_prox_data, 0, 1) * (pred_t2l_display.shape[0] - 1)

                    # Update predicted proximity traces if available
                    if pred_up_proximity is not None and pred_down_proximity is not None:
                        times = np.arange(len(pred_up_proximity))[: self._max_points]
                        pred_up_prox_data = pred_up_proximity[-self._max_points :]
                        pred_down_prox_data = pred_down_proximity[-self._max_points :]
                        self.up_proximity_trace.x = times
                        self.up_proximity_trace.y = np.clip(pred_up_prox_data, 0, 1) * (pred_t2l_display.shape[0] - 1)
                        self.down_proximity_trace.x = times
                        self.down_proximity_trace.y = np.clip(pred_down_prox_data, 0, 1) * (pred_t2l_display.shape[0] - 1)

                # Loss traces share x so both y-axes stay aligned.
                n_loss = min(len(self.losses), len(self.test_losses), self._loss_max_points)
                if n_loss:
                    loss_xs = list(range(n_loss))
                    self.loss_trace.x = loss_xs
                    self.loss_trace.y = self.losses[-n_loss:]
                    self.test_loss_trace.x = loss_xs
                    self.test_loss_trace.y = self.test_losses[-n_loss:]

                # PnL traces share x.
                if gt_pnl is not None and pred_pnl is not None:
                    n_pnl = min(len(gt_pnl), len(pred_pnl), self._max_points)
                    pnl_xs = list(range(n_pnl))
                    self.gt_pnl_trace.x = pnl_xs
                    self.gt_pnl_trace.y = list(gt_pnl[-n_pnl:])
                    self.pred_pnl_trace.x = pnl_xs
                    self.pred_pnl_trace.y = list(pred_pnl[-n_pnl:])

            self._refresh()

        except Exception as e:
            print(f"Error updating plot: {e}")

    def add_loss(self, train_loss: float, test_loss: float) -> None:
        """Append one (train_loss, test_loss) pair. Both lists are always the same length."""
        self.losses.append(float(train_loss))
        self.test_losses.append(float(test_loss))
        if len(self.losses) > self._max_points:
            self.losses = self.losses[-self._max_points :]
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
            # Match the saved dashboard rendering logic more closely:
            # non-negative clip + sqrt lift preserves sparse structure much
            # better than hard clipping into [-1, 1].
            t2l_data = np.sqrt(np.clip(t2l_array[:, :, 0].T, 0.0, None))

        return im_data, t2l_data, prices_array, feature_points


if __name__ == "__main__":
    vis = Visualizer()
    rng = np.random.default_rng(456456)
    data_dict = {
        "books_z_data": rng.random((10, 10, 3)),
        "level_reach_z_data": rng.random((10, 10, 1)),
        "bidask": rng.random((10, 2)),
    }
    vis.update(**data_dict)
