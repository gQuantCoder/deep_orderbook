# visu.py

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display


class Visualizer:
    """Visualizer class to encapsulate the figure and update methods."""

    def __init__(self):
        self.fig_widget = self._create_figure()
        self._initialize_traces()
        display(self.fig_widget)
        self.losses = []
        self.test_losses = []  # New list for test losses
        self._max_points = 1000  # Limit the number of points to store

    def _create_figure(self):
        """Creates and returns a Plotly figure widget with subplots."""
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
            # Add a secondary y-axis for test loss
            yaxis5=dict(
                title="Training Loss",
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue"),
                domain=[0.175, 0.3],
            ),
            yaxis6=dict(
                title="Test Loss",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                anchor="x5",
                overlaying="y5",
                side="right",
            ),
            yaxis7=dict(
                title="Ground Truth PnL",
                titlefont=dict(color="green"),
                tickfont=dict(color="green"),
                domain=[0.0, 0.16],
            ),
            yaxis8=dict(
                title="Predicted PnL",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                anchor="x6",
                overlaying="y7",
                side="right",
            ),
        )

        fig_widget = go.FigureWidget(fig)
        return fig_widget

    def _initialize_traces(self):
        """Initializes and adds traces to the figure widget."""
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

        # Heatmap for Books
        self.im_trace = go.Heatmap(
            z=np.zeros((10, 10)),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            showscale=False,
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

        # Line trace for Ground Truth PnL (left y-axis)
        self.gt_pnl_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="green", width=2),
            name="Ground Truth PnL",
            showlegend=True,
            yaxis="y7",
        )

        # Line trace for Predicted PnL (right y-axis)
        self.pred_pnl_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=2),
            name="Predicted PnL",
            showlegend=True,
            yaxis="y8",
        )

        # Add all traces to the figure widget
        self.fig_widget.add_trace(self.bid_trace, row=1, col=1)
        self.fig_widget.add_trace(self.ask_trace, row=1, col=1)
        self.fig_widget.add_trace(self.im_trace, row=2, col=1)
        self.fig_widget.add_trace(self.t2l_trace, row=3, col=1)
        self.fig_widget.add_trace(self.pred_trace, row=4, col=1)
        self.fig_widget.add_trace(self.loss_trace, row=5, col=1)
        self.fig_widget.add_trace(self.test_loss_trace, row=5, col=1)
        self.fig_widget.add_trace(self.gt_pnl_trace, row=6, col=1)  # PnL traces in row 6
        self.fig_widget.add_trace(self.pred_pnl_trace, row=6, col=1)

    def update(
        self,
        books_z_data: np.ndarray | None,
        level_reach_z_data: np.ndarray | None,
        bidask: np.ndarray | None,
        pred_t2l=None,
        gt_pnl: np.ndarray | None = None,
        pred_pnl: np.ndarray | None = None,
    ):
        """Updates the figure widget with new data."""
        try:
            with self.fig_widget.batch_update():
                books_z_data, level_reach_z_data, bidask = self.for_image_display(
                    books_z_data, level_reach_z_data, bidask
                )
                # Update bid and ask price traces with limited history
                if bidask is not None:
                    times = np.arange(min(bidask.shape[0], self._max_points))
                    bid_data = bidask[-self._max_points:, 0] if bidask.shape[0] > self._max_points else bidask[:, 0]
                    ask_data = bidask[-self._max_points:, 1] if bidask.shape[0] > self._max_points else bidask[:, 1]
                    
                    self.fig_widget.data[0].x = times
                    self.fig_widget.data[0].y = bid_data
                    self.fig_widget.data[1].x = times
                    self.fig_widget.data[1].y = ask_data

                # Update heatmaps
                if books_z_data is not None:
                    self.fig_widget.data[2].z = np.clip(books_z_data, -1, 1)
                if level_reach_z_data is not None:
                    self.fig_widget.data[3].z = np.clip(level_reach_z_data, -1, 1)

                # Update prediction heatmap
                if pred_t2l is not None:
                    self.fig_widget.data[4].z = np.clip(pred_t2l, -1, 1)

                # Update loss traces
                if self.losses:
                    loss_times = np.arange(len(self.losses))
                    self.fig_widget.data[5].x = loss_times
                    self.fig_widget.data[5].y = self.losses

                if self.test_losses:
                    test_loss_times = np.arange(len(self.test_losses))
                    self.fig_widget.data[6].x = test_loss_times
                    self.fig_widget.data[6].y = self.test_losses

                # Update PnL traces
                if gt_pnl is not None:
                    pnl_times = np.arange(len(gt_pnl))
                    self.fig_widget.data[7].x = pnl_times
                    self.fig_widget.data[7].y = gt_pnl
                    self.fig_widget.data[7].yaxis = "y7"

                if pred_pnl is not None:
                    pred_pnl_times = np.arange(len(pred_pnl))
                    self.fig_widget.data[8].x = pred_pnl_times
                    self.fig_widget.data[8].y = pred_pnl
                    self.fig_widget.data[8].yaxis = "y8"

        except Exception as e:
            print(f"Error updating plot: {e}")
        finally:
            # Force garbage collection after update
            import gc
            gc.collect()

    def add_loss(self, train_loss: float | None, test_loss: float):
        """Adds loss values to the loss history.
        
        Args:
            train_loss: Training loss value, can be None if only test loss is available
            test_loss: Test loss value
        """
        if train_loss is not None:
            self.losses.append(float(train_loss))  # Convert to float to ensure no reference holding
            # Keep only recent history
            if len(self.losses) > self._max_points:
                self.losses = self.losses[-self._max_points:]
        
        self.test_losses.append(float(test_loss))
        # Keep only recent history
        if len(self.test_losses) > self._max_points:
            self.test_losses = self.test_losses[-self._max_points:]

    @staticmethod
    def for_image_display(
        books_array: np.ndarray | None = None,
        t2l_array: np.ndarray | None = None,
        prices_array: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        im_data = None
        t2l_data = None
        if books_array is not None:
            im_data = books_array.copy()
            im_data[:, :, 0] *= -0.5
            im_data[:, :, 1:3] *= 1e6
            im_data = im_data.mean(axis=2).T
        if t2l_array is not None:
            t2l_data = t2l_array[:, :, 0].T

        return im_data, t2l_data, prices_array


if __name__ == '__main__':
    vis = Visualizer()
    data_dict = {
        "books_z_data": np.random.rand(10, 10, 3),
        "level_reach_z_data": np.random.rand(10, 10, 1),
        "bidask": np.random.rand(10, 2),
    }
    vis.update(**data_dict)
