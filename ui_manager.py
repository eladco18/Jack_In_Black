import numpy as np
from typing import Optional, Dict, Any
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QStackedWidget, QPushButton, QLineEdit, QComboBox,
    QProgressBar, QFrame, QDialog
)
from PyQt5.QtCore import pyqtSlot, Qt, pyqtSignal, QObject, QTimer, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QIntValidator, QColor, QPolygon
import os
import pygame


# --- 1. Communication Signals ---
# NOTE: The GameSignals class is defined in main.py and injected into the UIManager.
# Do not redefine it here to avoid dual-maintenance bugs.
class SetupDialog(QDialog):
    """
    Modal dialog for game initial setup.
    Gathers player names, bankrolls, game mode, and deck count.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Setup")

        # Block interaction with the main window until setup is done
        self.setModal(True)
        self.setFixedSize(800, 1000)

        # Dialog Styling
        self.setStyleSheet("""
            QDialog { background-color: #121212; border: 2px solid #333; }
            QLabel { font-size: 18px; font-weight: bold; color: #ddd; }
            QLineEdit, QComboBox { 
                font-size: 18px; padding: 8px; background-color: #2a2a2a; 
                color: white; border: 1px solid #444; border-radius: 6px; 
            }
            QPushButton { 
                font-size: 20px; font-weight: bold; padding: 12px; 
                background-color: #00e5ff; color: #000; border-radius: 6px; 
            }
            QPushButton:hover { background-color: #00b8cc; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        lbl_title = QLabel("SYSTEM SETUP")
        lbl_title.setStyleSheet("font-size: 28px; font-weight: bold; color: #00e5ff; margin-bottom: 10px;")
        layout.addWidget(lbl_title)

        self.input_p1 = QLineEdit()
        self.input_p1.setPlaceholderText("Player 1 Name")
        layout.addWidget(QLabel("Player 1 Name:"))
        layout.addWidget(self.input_p1)

        self.input_p2 = QLineEdit()
        self.input_p2.setPlaceholderText("Player 2 Name")
        layout.addWidget(QLabel("Player 2 Name:"))
        layout.addWidget(self.input_p2)

        integer_validator = QIntValidator(0, 3000)

        self.input_p1_bankroll = QLineEdit()
        self.input_p1_bankroll.setPlaceholderText("P1 Bankroll ($)")
        self.input_p1_bankroll.setText("250")
        self.input_p1_bankroll.setValidator(integer_validator)
        layout.addWidget(QLabel("Player 1 Initial Bankroll:"))
        layout.addWidget(self.input_p1_bankroll)

        self.input_p2_bankroll = QLineEdit()
        self.input_p2_bankroll.setPlaceholderText("P2 Bankroll ($)")
        self.input_p2_bankroll.setText("250")
        self.input_p2_bankroll.setValidator(integer_validator)
        layout.addWidget(QLabel("Player 2 Initial Bankroll:"))
        layout.addWidget(self.input_p2_bankroll)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Regular", "The Perfect Gambler", "Super Computer"])
        layout.addWidget(QLabel("Game Mode:"))
        layout.addWidget(self.combo_mode)

        self.combo_decks = QComboBox()
        self.combo_decks.addItems(["1", "2", "4", "6", "8"])
        self.combo_decks.setCurrentText("2")
        layout.addWidget(QLabel("Number of Decks:"))
        layout.addWidget(self.combo_decks)

        layout.addStretch()

        self.btn_start = QPushButton("Start Game")
        # Accept the dialog, which closes it and returns a success code
        self.btn_start.clicked.connect(self.accept)
        layout.addWidget(self.btn_start)

    def get_data(self) -> dict:
        """Returns the configured data as a dictionary."""
        return {
            "p1_name": self.input_p1.text() or "Player 1",
            "p2_name": self.input_p2.text() or "Player 2",
            "p1_bankroll": int(self.input_p1_bankroll.text() or 1000),
            "p2_bankroll": int(self.input_p2_bankroll.text() or 1000),
            "mode": self.combo_mode.currentText(),
            "num_decks": int(self.combo_decks.currentText())
        }


# --- 2. Custom Widgets ---
class CalibrationVideoWidget(QLabel):
    roi_selected = pyqtSignal(str, tuple)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        # self.setStyleSheet("background-color: #000; border: 2px solid #333;")
        self.setStyleSheet("background-color: #000; border: 4px solid #ececec; border-radius: 15px;")
        self.setMouseTracking(True)

        self.is_calibrating = False
        self.current_roi_name = ""
        self.origin_point = None
        self.current_rect = None
        self.last_frame = None
        self.original_resolution = (1280, 720)

    def set_calibration_mode(self, active: bool, target_name: str = ""):
        self.is_calibrating = active
        self.current_roi_name = target_name
        self.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)

    def update_frame(self, cv_img: np.ndarray):
        # Freeze the video feed ONLY when the user is actively clicking and dragging the mouse.
        # Otherwise, let the live feed run so the screen isn't black at startup!
        if self.is_calibrating and self.origin_point is not None:
            return

        self.original_resolution = (cv_img.shape[1], cv_img.shape[0])
        self.last_frame = cv_img

        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        if self.is_calibrating and event.button() == Qt.LeftButton:
            self.origin_point = event.pos()
            self.current_rect = QRect(self.origin_point, self.origin_point)
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_calibrating and self.origin_point:
            self.current_rect = QRect(self.origin_point, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.is_calibrating and event.button() == Qt.LeftButton and self.current_rect:
            pixmap = self.pixmap()
            if not pixmap: return

            x_offset = (self.width() - pixmap.width()) // 2
            y_offset = (self.height() - pixmap.height()) // 2

            x = self.current_rect.x() - x_offset
            y = self.current_rect.y() - y_offset
            w = self.current_rect.width()
            h = self.current_rect.height()

            scale_x = self.original_resolution[0] / pixmap.width()
            scale_y = self.original_resolution[1] / pixmap.height()

            final_roi = (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))

            if final_roi[2] > 5 and final_roi[3] > 5:
                self.roi_selected.emit(self.current_roi_name, final_roi)

            self.origin_point = None
            self.current_rect = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.is_calibrating and self.current_rect:
            painter = QPainter(self)
            pen = QPen(Qt.red, 3, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)


class DealerForecastWidget(QFrame):
    """
    Standalone widget for displaying the dealer's probability forecast.
    Placed under the video feed in Super Computer mode.
    """

    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
                    background-color: #4a4a4a; 
                    border-radius: 15px; 
                    padding: 1px; 
                    border: 5px solid #00e5ff; 
                    margin: 1px;
                """)
        layout = QVBoxLayout(self)
        self.setMinimumWidth(700)
        self.setMinimumHeight(250)

        title_lbl = QLabel("🎯 Dealer Forecast 🎯")
        title_lbl.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 48px; border: none;")
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)
        layout.setSpacing(2)

        self.dealer_probs_labels = {}
        probs_layout = QHBoxLayout()

        probs_layout.setSpacing(10)  # Adjust horizontal gap between numbers
        prob_keys = ["17", "18", "19", "20", "21", "Bust"]
        colors = {"Bust": "#f44336", "21": "#5bc85f", "DEFAULT": "#ffeb3b"}

        for key in prob_keys:
            color = colors.get(key, colors["DEFAULT"])
            item_layout = QVBoxLayout()
            item_layout.setSpacing(2)

            lbl_title = QLabel(str(key))
            lbl_title.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 28px; border: none;")
            lbl_title.setAlignment(Qt.AlignCenter)

            lbl_val = QLabel("0.0%")
            lbl_val.setStyleSheet("color: white; font-size: 28px; border: none;")
            lbl_val.setAlignment(Qt.AlignCenter)

            item_layout.addWidget(lbl_title)
            item_layout.addWidget(lbl_val)
            probs_layout.addLayout(item_layout)

            # Save reference to update the text later
            self.dealer_probs_labels[key] = lbl_val

        layout.addLayout(probs_layout)

    def update_forecast(self, d_probs: Optional[Dict[Any, float]]):
        """Updates the probabilities dynamically based on engine output."""
        if not d_probs:
            return

        for key, lbl in self.dealer_probs_labels.items():
            # Convert string key back to int if it's a number (e.g. "17" -> 17)
            dict_key = int(key) if key.isdigit() else key
            val = d_probs.get(dict_key, 0.0)
            lbl.setText(f"{val:.1f}%")


class BetSpeedometer(QWidget):
    """
    Custom widget that draws a semi-circular speedometer gauge to display Expected Value (EV).
    Left side represents casino advantage, right side represents player advantage.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # 1. Increased the minimum height slightly to give the gauge safe breathing room
        self.setMinimumSize(250, 130)
        self.ev_value = -0.5  # Default EV value (percentage)

    def set_ev(self, ev):
        self.ev_value = ev
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 2. Shifted the drawing area up (y=10 instead of 25)
        # The center of this rect is exactly at (125, 105)
        rect = QRect(30, 10, 190, 190)

        # Define the pen style only once!
        # This ensures smooth, flat connections between the colors
        pen = QPen(Qt.SolidLine)
        pen.setWidth(18)
        pen.setCapStyle(Qt.FlatCap)

        # 3. Change *only the color* before each arc, so we don't lose the styling

        # Red arc (Casino edge) - from 180 degrees downwards for 60 degrees
        pen.setColor(QColor("#ff5555"))
        painter.setPen(pen)
        painter.drawArc(rect, 180 * 16, -60 * 16)

        # Orange arc (Neutral) - from 120 degrees downwards for 60 degrees
        pen.setColor(QColor("#ff9800"))
        painter.setPen(pen)
        painter.drawArc(rect, 120 * 16, -60 * 16)

        # Green arc (Player edge) - from 60 degrees downwards for 60 degrees
        pen.setColor(QColor("#39ff14"))
        painter.setPen(pen)
        painter.drawArc(rect, 60 * 16, -60 * 16)

        clamped_ev = max(-2, min(2, self.ev_value))
        angle = 90 - (clamped_ev * 45)

        # 4. Move the needle's rotation pivot exactly to the center of our new QRect
        painter.translate(125, 105)
        painter.rotate(-angle)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#ffffff"))

        # Draw the needle
        needle = [QPoint(0, -5), QPoint(0, 5), QPoint(80, 0)]
        painter.drawPolygon(QPolygon(needle))

        # Draw the center pivot point (the screw)
        painter.setBrush(QColor("#444"))
        painter.drawEllipse(QPoint(0, 0), 8, 8)


class BetRecommendationWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
                    background-color: #4a4a4a; 
                    border-radius: 15px; 
                    padding: 1px; 
                    border: 5px solid #00e5ff; 
                    margin: 1px;
                """)

        # 1. Expand the cube's width
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        layout.setSpacing(2)  # Tighter layout

        self.lbl_rec_title = QLabel("ROUND EVALUATION")
        self.lbl_rec_title.setStyleSheet("color: #00e5ff; font-size: 24px; font-weight: bold; border: none; "
                                         "margin-bottom: 0px;")
        self.lbl_rec_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_rec_title)

        self.lbl_bet_rec = QLabel("MINIMUM BET")
        self.lbl_bet_rec.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_bet_rec)

        self.speedo = BetSpeedometer()
        layout.addWidget(self.speedo, 0, Qt.AlignCenter)

        # 2. Add a stretch (virtual spring) to push everything below it to the bottom of the widget
        layout.addStretch()

        self.stats_layout = QHBoxLayout()
        # Remove bottom margins from the horizontal layout, so it sticks to the bottom edge
        self.stats_layout.setContentsMargins(0, 5, 0, 0)

        self.lbl_win = QLabel("WIN: 0%")
        self.lbl_tie = QLabel("TIE: 0%")
        self.lbl_loss = QLabel("LOSE: 0%")

        self.lbl_win.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 22px; border: none;")
        self.lbl_tie.setStyleSheet("color: #ffeb3b; font-weight: bold; font-size: 22px; border: none;")
        self.lbl_loss.setStyleSheet("color: #f44336; font-weight: bold; font-size: 22px; border: none;")

        self.lbl_win.setAlignment(Qt.AlignCenter)
        self.lbl_tie.setAlignment(Qt.AlignCenter)
        self.lbl_loss.setAlignment(Qt.AlignCenter)

        self.stats_layout.addWidget(self.lbl_win)
        self.stats_layout.addWidget(self.lbl_tie)
        self.stats_layout.addWidget(self.lbl_loss)

        layout.addLayout(self.stats_layout)

    def update_data(self, bet_rec: str, p_stats: dict, ev: float = 0.0):
        self.lbl_bet_rec.setText(str(bet_rec).upper())
        self.speedo.set_ev(ev)

        # Reduced dynamic text font sizes from 26px to 18px
        if "HIGH" in bet_rec.upper() or "INCREASE" in bet_rec.upper():
            self.lbl_bet_rec.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 20px; border: none;")
        elif "MINIMUM" in bet_rec.upper():
            self.lbl_bet_rec.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 20px; border: none;")
        else:
            self.lbl_bet_rec.setStyleSheet("color: #ecf0f1; font-weight: bold; font-size: 20px; border: none;")

        win_pct = p_stats.get("win", 0) * 100
        tie_pct = p_stats.get("tie", 0) * 100
        loss_pct = p_stats.get("loss", 0) * 100

        self.lbl_win.setText(f"WIN: {win_pct:.1f}%")
        self.lbl_tie.setText(f"TIE: {tie_pct:.1f}%")
        self.lbl_loss.setText(f"LOSE: {loss_pct:.1f}%")


class ProbabilityDonut(QWidget):
    """
    Custom widget that draws a micro donut chart for Win/Loss/Tie probabilities.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(60, 60)
        self.win = 0.0
        self.loss = 0.0
        self.tie = 0.0

    def set_values(self, win, loss, tie):
        self.win = win
        self.loss = loss
        self.tie = tie
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(5, 5, -5, -5)
        thickness = 14

        # Draw background/Tie (Gray)
        painter.setPen(QPen(Qt.gray, thickness, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 0, 360 * 16)

        # Calculate angles (1 unit = 1/16th of a degree in Qt)
        win_angle = int(self.win * 360 * 16)
        loss_angle = int(self.loss * 360 * 16)

        # Draw Loss (Red)
        if self.loss > 0:
            painter.setPen(QPen(QColor("#ff3333"), thickness, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(rect, 90 * 16, -loss_angle)

        # Draw Win (Neon Blue/Green)
        if self.win > 0:
            painter.setPen(QPen(QColor("#39ff14"), thickness, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(rect, 90 * 16, win_angle)


class StrategyWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
                    background-color: #3b5947;
                    border-radius: 15px;
                    padding: 10px;
                    border: 3px solid #2a4033;
                """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)

        # Title
        self.lbl_title = QLabel("AI ANALYSIS")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet(
            "font-size: 16px; color: #888; font-weight: bold; letter-spacing: 2px; "
            "border: none; margin-bottom: 5px;")
        self.main_layout.addWidget(self.lbl_title)

        # Container for the dynamic action columns (Horizontal Layout)
        self.rows_container = QWidget()
        # --- MODIFIED: Changed to QHBoxLayout to align action columns horizontally ---
        self.rows_layout = QHBoxLayout(self.rows_container)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(15)
        self.rows_layout.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.rows_container)

    def update_recommendation(self, best_action: str, stats: Optional[Dict[str, Any]]):
        """
        Updates the widget. Dynamically creates a vertical column (Action, Donut, EV)
        for EVERY evaluated action and lays them out horizontally.
        """
        # 1. Clear previous columns
        for i in reversed(range(self.rows_layout.count())):
            widget = self.rows_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        color_map = {
            "HIT": "#00e5ff", "STAND": "#ffeb3b", "DOUBLE": "#ff9800",
            "SURRENDER": "#9e9e9e", "BUST": "#ff5555", "ERROR": "#ff5555"
        }

        # 2. Check if we have Super Computer detailed stats
        if stats and "all_actions" in stats and stats["all_actions"]:
            all_acts = stats["all_actions"]

            # Sort actions by EV (highest to lowest)
            sorted_actions = sorted(all_acts.items(), key=lambda item: item[1].get('ev', -float('inf')), reverse=True)

            for action_name, act_data in sorted_actions:
                # --- Create a vertical column for this specific action ---
                col_widget = QWidget()
                col_layout = QVBoxLayout(col_widget)
                col_layout.setContentsMargins(10, 10, 10, 10)
                col_widget.setMinimumWidth(240)
                self.rows_layout.setSpacing(35)
                col_widget.setStyleSheet("background-color: transparent; border: none;")
                col_layout.setAlignment(Qt.AlignCenter)

                # A. Action Name Label
                lbl_act = QLabel(action_name)
                lbl_act.setAlignment(Qt.AlignCenter)
                color = color_map.get(action_name, "#ffffff")

                # B. Donut Chart
                donut = ProbabilityDonut()
                s = act_data.get('stats', {})
                donut.set_values(s.get('win', 0.0), s.get('loss', 0.0), s.get('tie', 0.0))

                # C. EV Label
                ev = act_data.get('ev', 0.0)
                lbl_ev = QLabel(f"EV: {ev * 100:+.1f}%")  # EV as Percentage
                lbl_ev.setAlignment(Qt.AlignCenter)

                # --- Glowing Highlight for Best Action ---
                donut.setFixedSize(80, 80)
                lbl_act.setStyleSheet(
                    f"color: {color}; font-weight: bold; font-size: 24px; opacity: 0.6; border: none;")

                if action_name == best_action:
                    col_widget.setStyleSheet(f"""
                                        background-color: rgba(0, 229, 255, 0.08); 
                                        border: 5px solid {color}; 
                                        border-radius: 12px;
                                    """)
                    if ev > 0:
                        lbl_ev.setStyleSheet("font-size: 24px; font-weight: bold; color: #39ff14; border: none;")
                    elif ev < 0:
                        lbl_ev.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff5555; border: none;")
                    else:
                        lbl_ev.setStyleSheet("font-size: 24px; font-weight: bold; color: #aaa; border: none;")
                else:
                    col_widget.setStyleSheet("background-color: transparent; border: 2px solid transparent;")
                    lbl_act.setStyleSheet(
                        f"color: {color}; font-weight: bold; font-size: 24px; opacity: 0.5; border: none;")

                    if ev > 0:
                        lbl_ev.setStyleSheet(
                            "font-size: 22px; font-weight: bold; color: #39ff14; opacity: 0.5; border: none;")
                    elif ev < 0:
                        lbl_ev.setStyleSheet(
                            "font-size: 22px; font-weight: bold; color: #ff5555; opacity: 0.5; border: none;")
                    else:
                        lbl_ev.setStyleSheet(
                            "font-size: 22px; font-weight: bold; color: #aaa; opacity: 0.5; border: none;")

                # Assemble the vertical column
                col_layout.addWidget(lbl_act)
                col_layout.addWidget(donut, 0, Qt.AlignCenter)

                # Add a virtual spring to push everything below it to the absolute bottom
                col_layout.addStretch()

                # Add the EV label and explicitly align it to the bottom-center
                col_layout.addWidget(lbl_ev, 0, Qt.AlignBottom | Qt.AlignHCenter)

                # Add the compiled column to the main horizontal container
                self.rows_layout.addWidget(col_widget)

        else:
            # Fallback for Perfect Gambler Mode (No stats, just the action)
            if best_action == "HIT":
                display_text = "HIT 🔵"
            elif best_action == "STAND":
                display_text = "STAND 🔴"
            elif best_action == "DOUBLE":
                display_text = "DOUBLE 🟡"
            elif best_action == "SURRENDER":
                display_text = "SURRENDER 🟤"
            elif best_action == "BUST":
                display_text = "BUST 💥"
            else:
                display_text = best_action if best_action else "--"

            lbl_act = QLabel(display_text)
            color = color_map.get(best_action, "#ffffff")

            lbl_act.setStyleSheet(f"""
                            color: {color}; 
                            font-weight: 900; 
                            font-size: 60px; 
                            border: none; 
                            background: transparent;
                            letter-spacing: 5px;
                        """)
            lbl_act.setAlignment(Qt.AlignCenter)

            # Add a bit of breathing room to the parent container
            self.setContentsMargins(20, 30, 20, 30)
            self.rows_layout.addWidget(lbl_act)


class FinancialResultWidget(QFrame):
    """
    A dedicated pop-up widget that displays the net profit/loss
    at the end of a round. Completely decoupled from game modes.
    """

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #1a1a1a; border-radius: 10px; border: 2px solid #333; padding: 10px;")
        self.setMinimumHeight(120)
        self.setMinimumWidth(350)
        self.hide()  # Hidden by default

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.lbl_title = QLabel("ROUND RESULT")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet(
            "color: #ccc; font-size: 20px; font-weight: bold; border: none; letter-spacing: 2px;")

        self.lbl_amount = QLabel("--")
        self.lbl_amount.setAlignment(Qt.AlignCenter)
        self.lbl_amount.setStyleSheet("font-size: 50px; font-weight: 900; border: none;")

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_amount)

    def show_result(self, payout: int, bet: int):
        """Formats and colors the result based on Gross Payout logic."""
        self.lbl_amount.setText(f"+{payout} $" if payout > 0 else "0 $")

        # Color logic based on whether the player is "Up", "Even", or "Down"
        if payout > bet:  # WIN
            self.setStyleSheet(
                "background-color: #1a1a1a; border-radius: 10px; border: 3px solid #39ff14; padding: 10px;")
            self.lbl_amount.setStyleSheet("color: #39ff14; font-size: 50px; font-weight: 900; border: none;")
        elif payout == bet:  # PUSH
            self.setStyleSheet("background-color: #1a1a1a; border-radius: 10px; border: 3px solid #888; padding: 10px;")
            self.lbl_amount.setStyleSheet("color: #fff; font-size: 50px; font-weight: 900; border: none;")
        else:  # LOSS or SURRENDER
            self.setStyleSheet(
                "background-color: #1a1a1a; border-radius: 10px; border: 3px solid #ff5555; padding: 10px;")
            self.lbl_amount.setStyleSheet("color: #ff5555; font-size: 50px; font-weight: 900; border: none;")

        self.show()


class HouseRulesWidget(QFrame):
    """
    Displays the static and dynamic rules of the current game session.
    """

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #4a4a4a; border-radius: 15px; padding: 1px; border: 5px solid #f4c430;"
                           "margin: 1px; border-radius: 15px")
        self.setFixedSize(350, 260)
        layout = QVBoxLayout(self)

        # Set spacing between rows to absolute 0
        layout.setSpacing(0)

        title = QLabel("HOUSE RULES")
        title.setStyleSheet("color: #f4c430; font-size: 30px; font-weight: bold; border: none; margin-bottom: 6px;")
        layout.addWidget(title)

        # Labels for dynamic updates
        self.lbl_s17 = QLabel("• Dealer Stands on Soft 17")
        self.lbl_decks = QLabel("• Decks in Shoe: --")
        self.lbl_cut = QLabel("• Shuffle at: -- cards")
        self.lbl_payout = QLabel("• Blackjack Pays: 3:2")

        for lbl in [self.lbl_s17, self.lbl_decks, self.lbl_cut, self.lbl_payout]:
            lbl.setStyleSheet("color: #bbb; font-size: 23px; border: none;")
            layout.addWidget(lbl)

    def update_rules(self, num_decks: int, threshold: int):
        """Updates the dynamic part of the rules."""
        self.lbl_decks.setText(f"• Decks in Shoe: {num_decks}")
        self.lbl_cut.setText(f"• Shuffle at: {threshold} cards")


# --- 3. Main UI Manager ---
class UIManager(QMainWindow):
    def __init__(self, signals: QObject, enable_color_norm: bool = False):
        super().__init__()
        self.signals = signals
        self.enable_color_norm = enable_color_norm

        self.rois = {}
        self.p1_name_val = "Player 1"
        self.p2_name_val = "Player 2"

        self.setWindowTitle("Blackjack AI Analyzer | Professional Edition")
        self.resize(1400, 800)

        # --- Audio Initialization ---
        pygame.mixer.init()
        self.sounds = {}
        self._load_sounds()

        self.central_widget = QWidget()
        self.central_widget.setObjectName("centralWidget")
        self.setCentralWidget(self.central_widget)

        self.setAutoFillBackground(True)  # forces the main window to paint the palette

        # Apply the rest of the styles (Buttons, text, etc.)
        self._apply_styles()

        # --- Main Layout Foundation (Vertical Base) ---
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # ========================================================
        # 1. TOP BANNER
        # ========================================================
        self.top_banner_frame = QFrame()
        self.top_banner_frame.setFixedHeight(400)

        self.top_horizontal_layout = QHBoxLayout(self.top_banner_frame)
        self.top_horizontal_layout.setContentsMargins(15, 10, 15, 10)

        self.top_p1_strat_layout = QVBoxLayout()
        self.top_alerts_layout = QVBoxLayout()
        self.top_dealer_layout = QVBoxLayout()
        self.top_p2_strat_layout = QVBoxLayout()

        self.top_horizontal_layout.addLayout(self.top_p1_strat_layout, stretch=5)
        self.top_horizontal_layout.addLayout(self.top_alerts_layout, stretch=2)
        self.top_horizontal_layout.addLayout(self.top_dealer_layout, stretch=2)
        self.top_horizontal_layout.addLayout(self.top_p2_strat_layout, stretch=5)

        self.lbl_calib_title = QLabel("CALIBRATION MODE")
        self.lbl_calib_title.setStyleSheet("font-size: 36px; font-weight: bold; color: #ffeb3b;")
        self.lbl_calib_title.setAlignment(Qt.AlignCenter)

        self.lbl_instruction = QLabel("Initializing...")
        self.lbl_instruction.setStyleSheet("font-size: 36px; font-weight: bold; margin: 20px 0; color: #4caf50")
        self.lbl_instruction.setWordWrap(True)
        self.lbl_instruction.setAlignment(Qt.AlignCenter)

        self.top_alerts_layout.addWidget(self.lbl_calib_title)
        self.top_alerts_layout.addWidget(self.lbl_instruction)

        self.main_layout.addWidget(self.top_banner_frame)

        self.base_bottom_cube = "background-color: #1a1a2e; border: 3px solid #ffffff; border-radius: 10px; " \
                                "padding: 5px; margin-bottom: 5px; "

        # ========================================================
        # 2. MIDDLE SECTION (3 Columns: P1, Video, P2)
        # ========================================================
        self.columns_layout = QHBoxLayout()
        self.main_layout.addLayout(self.columns_layout, stretch=1)

        # --- LEFT COLUMN (Player 1) ---
        self.left_column = QVBoxLayout()
        self.left_column.setContentsMargins(10, 10, 10, 10)

        # --- CENTER COLUMN (Video Feed) ---
        self.center_column = QVBoxLayout()
        self.center_column.setContentsMargins(10, 10, 10, 10)
        self.center_column.setSpacing(10)
        self.video_widget = CalibrationVideoWidget()
        self.video_widget.roi_selected.connect(self._on_roi_captured)
        self.center_column.addWidget(self.video_widget, stretch=1)

        # --- RIGHT COLUMN (Player 2) ---
        self.right_column = QVBoxLayout()
        self.right_column.setContentsMargins(10, 10, 10, 10)

        self.control_panel = QStackedWidget()
        self.right_column.addWidget(self.control_panel)

        self.columns_layout.addLayout(self.left_column, stretch=2)
        self.columns_layout.addLayout(self.center_column, stretch=8)
        self.columns_layout.addLayout(self.right_column, stretch=2)

        # ========================================================
        # Bottom HUD is created in _init_game_dashboard
        # ========================================================

        self._init_calibration_page()
        self._init_game_dashboard()

        self.control_panel.setCurrentIndex(0)

        self._decision_remaining = 0
        self._decision_total = 0
        self._decision_label = "Waiting for decision"

        self._connect_signals()
        self.start_calibration_process()

    def _load_sounds(self):
        """Pre-loads all sound files into memory for instant, zero-latency playback."""
        sound_files = {
            "tick": "tick.wav",
            "deal": "dealing.wav",
            "win": "win.wav",
            "lose": "lose.wav",
            "blackjack": "blackjack.wav",
            "push": "push.wav",
            "chips": "chips.wav",
            "shuffle": "shuffle.wav",
            "new_card": "new card.wav",
            "money_drop": "money drop.wav",
            "no_more_bets": "no more bets.wav",
            "all in": "all in.wav"
        }

        for key, filename in sound_files.items():
            full_path = os.path.join("sounds", filename)
            if os.path.exists(full_path):
                self.sounds[key] = pygame.mixer.Sound(full_path)
            else:
                print(f"[UI MANAGER] Warning: Audio file '{full_path}' not found.")
                self.sounds[key] = None

    def play_sound(self, name: str):
        """Safely plays a sound asynchronously if it was successfully loaded."""
        if self.sounds.get(name):
            self.sounds[name].play()

    def stop_sound(self, name: str):
        """Safely stops a sound if it is currently playing."""
        if self.sounds.get(name):
            self.sounds[name].stop()

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { 
                background-color: qradialgradient(cx: 0.5, cy: 0.5, radius: 2, fx: 0.5, fy: 0.5, 
                                  stop: 0 #1f2235, stop: 1 #0a0a10); 
            }
            QLabel { color: #ffffff; font-family: 'Roboto', sans-serif; }
            QLineEdit, QComboBox { 
                background-color: #2a2a2a; color: white; border: 1px solid #444; 
                padding: 10px; border-radius: 6px; font-size: 14px;
            }
            QPushButton {
                background-color: #00e5ff; color: #000; font-weight: bold;
                border-radius: 6px; padding: 20px; font-size: 22px; text-transform: uppercase;
            }
            QPushButton:hover { background-color: #00b8cc; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """)

    def show_setup_dialog(self):
        """Shows the popup dialog for game configuration."""
        dialog = SetupDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            setup_data = dialog.get_data()
            self._start_game_with_data(setup_data)

    def _init_calibration_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.btn_confirm_roi = QPushButton("Confirm Selection")
        self.btn_confirm_roi.setStyleSheet(
            "background-color: #444; color: #888; font-weight: bold; padding: 12px; border-radius: 6px;")
        self.btn_confirm_roi.setEnabled(False)
        self.btn_confirm_roi.clicked.connect(self._advance_calibration)

        self.required_rois = []
        if self.enable_color_norm:
            self.required_rois.append("Felt Calibration")

        self.required_rois.extend([
            "Dealer Cards", "Player 1 Cards", "Player 2 Cards",
            "Player 1 Chips", "Player 2 Chips", "P1 Decision", "P2 Decision"
        ])

        self.roi_index = 0

        # Place the confirm button directly in the calibration page layout
        layout.addWidget(self.btn_confirm_roi)
        self.control_panel.addWidget(page)

    def _init_game_dashboard(self):
        # ========================================================
        # --- Top Banner Elements ---
        # ========================================================
        self.p1_strategy_display = StrategyWidget()
        self.top_p1_strat_layout.addWidget(self.p1_strategy_display)
        self.p1_strategy_display.hide()

        # --- P1 Financial Result placed in TOP layout, aligned Left ---
        self.p1_result_widget = FinancialResultWidget()
        self.top_p1_strat_layout.addWidget(self.p1_result_widget, alignment=Qt.AlignTop | Qt.AlignLeft)

        self.alert_box = QLabel("")
        self.alert_box.setAlignment(Qt.AlignCenter)
        self.alert_box.setWordWrap(True)
        self.top_alerts_layout.addWidget(self.alert_box)
        self.alert_box.hide()

        self.lbl_timer = QLabel("Waiting")
        self.lbl_timer.setAlignment(Qt.AlignCenter)
        self.lbl_timer.setStyleSheet(self.base_bottom_cube + "font-size: 28px; color: #aaa; font-weight: bold;")
        self.top_alerts_layout.addWidget(self.lbl_timer)
        self.lbl_timer.hide()

        self.timer_bar = QProgressBar()
        self.timer_bar.setTextVisible(False)
        self.timer_bar.setFixedHeight(40)  # Timer progress bar width
        self.timer_bar.setStyleSheet("""
                    QProgressBar { background-color: #333; border: none; border-radius: 4px; }
                    QProgressBar::chunk { background-color: #ffeb3b; border-radius: 4px; }
                """)
        self.top_alerts_layout.addWidget(self.timer_bar)
        self.timer_bar.hide()

        self.dealer_frame = QFrame()
        self.dealer_frame.setStyleSheet("background-color: #1e1e1e; border-radius: 8px; border: 1px solid #333;")
        dealer_layout = QVBoxLayout(self.dealer_frame)

        self.lbl_dealer_title = QLabel("DEALER")
        self.lbl_dealer_title.setAlignment(Qt.AlignCenter)
        self.lbl_dealer_title.setStyleSheet("color: #ff9800; font-size: 50px; font-weight: bold; letter-spacing: 3px;"
                                            "border: none;")

        self.lbl_dealer_score = QLabel("--")
        self.lbl_dealer_score.setAlignment(Qt.AlignCenter)
        self.lbl_dealer_score.setStyleSheet("font-size: 70px; font-weight: bold; color: #fff;")

        dealer_layout.addWidget(self.lbl_dealer_title)
        dealer_layout.addWidget(self.lbl_dealer_score)
        self.top_dealer_layout.addWidget(self.dealer_frame)
        self.dealer_frame.hide()

        self.p2_strategy_display = StrategyWidget()
        self.top_p2_strat_layout.addWidget(self.p2_strategy_display)
        self.p2_strategy_display.hide()

        # --- P2 Financial Result placed in TOP layout, aligned Right ---
        self.p2_result_widget = FinancialResultWidget()
        self.top_p2_strat_layout.addWidget(self.p2_result_widget, alignment=Qt.AlignTop | Qt.AlignRight)

        # ========================================================
        # --- Player Widgets ---
        # ========================================================
        self.p1_widget = self._create_player_widget("Player 1", "#00e5ff")
        self.lbl_p1_name = self.p1_widget.findChild(QLabel, "name")
        self.lbl_p1_hand = self.p1_widget.findChild(QLabel, "hand")
        self.lbl_p1_status = self.p1_widget.findChild(QLabel, "status")
        self.lbl_p1_prediction = self.p1_widget.findChild(QLabel, "prediction")
        self.lbl_p1_bet = self.p1_widget.findChild(QLabel, "bet")
        self.lbl_p1_bank = self.p1_widget.findChild(QLabel, "bank")
        self.p1_widget.hide()
        self.left_column.addWidget(self.p1_widget)

        self.p2_widget = self._create_player_widget("Player 2", "#bd00ff")
        self.lbl_p2_name = self.p2_widget.findChild(QLabel, "name")
        self.lbl_p2_hand = self.p2_widget.findChild(QLabel, "hand")
        self.lbl_p2_status = self.p2_widget.findChild(QLabel, "status")
        self.lbl_p2_prediction = self.p2_widget.findChild(QLabel, "prediction")
        self.lbl_p2_bet = self.p2_widget.findChild(QLabel, "bet")
        self.lbl_p2_bank = self.p2_widget.findChild(QLabel, "bank")
        self.p2_widget.hide()
        self.right_column.addWidget(self.p2_widget)

        # ========================================================
        # --- THE NEW BOTTOM HUD (Row-Based, Full Width) ---
        # ========================================================
        self.bottom_hud = QFrame()
        self.bottom_hud.setFixedHeight(350)
        # self.bottom_hud.setStyleSheet("background-color: #121212; border-top: 1px solid #333;")  # bottom background
        bottom_hud_layout = QHBoxLayout(self.bottom_hud)
        bottom_hud_layout.setContentsMargins(0, 10, 20, 10)

        # 1. FAR LEFT: Logo
        self.lbl_logo = QLabel()
        logo_path = os.path.join("pictures", "logo.png")
        logo_pixmap = QPixmap(logo_path)
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_logo.setPixmap(logo_pixmap)
        else:
            self.lbl_logo.setText("BLACKJACK\nPRO")
            self.lbl_logo.setStyleSheet("color: #FFD700; font-size: 22px; font-weight: bold;")

        # Align logo to the left (Sits under Player 1)
        bottom_hud_layout.addWidget(self.lbl_logo, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        bottom_hud_layout.addStretch(1)  # Pushes the center widget to the exact middle

        # 2. CENTER: Forecast & Bet Recommendation (Sits under Video)
        self.center_analysis_container = QWidget()
        analysis_layout = QHBoxLayout(self.center_analysis_container)
        analysis_layout.setSpacing(30)

        self.bet_recommendation_widget = BetRecommendationWidget()
        self.bet_recommendation_widget.hide()
        self.dealer_forecast_widget = DealerForecastWidget()
        self.dealer_forecast_widget.hide()

        analysis_layout.addWidget(self.bet_recommendation_widget)
        analysis_layout.addWidget(self.dealer_forecast_widget)

        bottom_hud_layout.addWidget(self.center_analysis_container, alignment=Qt.AlignCenter)

        bottom_hud_layout.addStretch(1)  # Pushes the right widgets to the edge

        # 3. FAR RIGHT: Buttons & House Rules (Sits under Player 2)
        self.right_controls_container = QWidget()
        right_layout = QHBoxLayout(self.right_controls_container)
        right_layout.setSpacing(20)

        # Remove the default invisible margins PyQt adds to layouts
        right_layout.setContentsMargins(0, 0, 0, 0)

        # A. Buttons Container
        self.btn_container = QWidget()
        btn_layout = QVBoxLayout(self.btn_container)
        btn_layout.setSpacing(10)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_new_round = QPushButton("New Round")
        self.btn_new_round.setStyleSheet(
            "background-color: #00e5ff; color: #000; font-weight: bold; padding: 12px; font-size: 18px; "
            "border-radius: 6px;")
        self.btn_new_round.clicked.connect(self._on_new_round_btn_pressed)

        self.btn_new_game = QPushButton("New Game")
        self.btn_new_game.setStyleSheet(
            "background-color: #00e5ff; color: #000; font-weight: bold; padding: 12px; font-size: 18px; "
            "border-radius: 6px;")
        self.btn_new_game.clicked.connect(self._on_new_game_btn_pressed)

        self.btn_shuffle_done = QPushButton("Shuffle Complete")
        self.btn_shuffle_done.setStyleSheet(
            "background-color: #00e5ff; color: #000; font-weight: bold; padding: 12px; font-size: 18px; "
            "border-radius: 6px;")
        self.btn_shuffle_done.clicked.connect(self._on_shuffle_done_pressed)
        self.btn_shuffle_done.hide()

        btn_layout.addWidget(self.btn_new_round)
        btn_layout.addWidget(self.btn_new_game)
        btn_layout.addWidget(self.btn_shuffle_done)
        self.btn_container.hide()

        right_layout.addWidget(self.btn_container)

        # B. House Rules Widget
        self.house_rules_widget = HouseRulesWidget()
        right_layout.addWidget(self.house_rules_widget)

        bottom_hud_layout.addWidget(self.right_controls_container, alignment=Qt.AlignRight | Qt.AlignVCenter)

        # FINALLY: Add the Bottom HUD to the very bottom of the Main Layout
        self.main_layout.addWidget(self.bottom_hud)

    def _create_player_widget(self, default_name, accent_color):
        """
        Creates a stylish player card widget with a consistent 'cube' layout
        for Status, Hand, Bet, and Bankroll.
        """
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{ background-color: #1a1a1a; border-radius: 10px; border: 2px solid {accent_color}; }}
        """)
        layout = QVBoxLayout(frame)
        layout.setSpacing(10)

        # 1. Player Name
        lbl_name = QLabel(default_name)
        lbl_name.setObjectName("name")
        lbl_name.setAlignment(Qt.AlignCenter)
        lbl_name.setStyleSheet(
            f"font-size: 36px; font-weight: bold; color: {accent_color}; margin-bottom: 5px; border: none;")
        layout.addWidget(lbl_name)

        # Helper function to generate consistent UI cubes
        def create_cube(title_text, obj_name, default_val, font_size="36px", cube_height=90):
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setSpacing(0)  # Zero spacing between title and cube
            vbox.setContentsMargins(0, 0, 0, 0)

            lbl_title = QLabel(title_text)
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #ccc; border: none; letter-spacing: 1px;")

            lbl_val = QLabel(default_val)
            lbl_val.setObjectName(obj_name)
            lbl_val.setAlignment(Qt.AlignCenter)
            lbl_val.setFixedHeight(100)  # Fixed height for the cube look
            lbl_val.setFixedHeight(cube_height)

            lbl_val.setStyleSheet(f"background-color: #1c1c1c; border: none; border-radius: 8px; color: "
                                  f"#444; font-size: {font_size}; font-weight: bold;")

            vbox.addWidget(lbl_title)
            vbox.addWidget(lbl_val)
            return container, lbl_val

        # 2. Hand Cube
        hand_container, lbl_hand = create_cube("CURRENT HAND 🃏", "hand", "--", "70px", cube_height=140)

        layout.addWidget(hand_container)

        # 3. Status Cube
        status_container, lbl_status = create_cube("CURRENT STATUS", "status", "--", "28px")
        layout.addWidget(status_container)

        # 4. Win Prediction Cube (Now a permanent cube)
        predict_container, lbl_prediction = create_cube("WIN PROBABILITY 📊", "prediction", "--", "28px")
        layout.addWidget(predict_container)

        # We hide the container by default; it will be shown only in Super Computer mode
        predict_container.setObjectName(f"predict_container_{default_name.replace(' ', '_')}")
        predict_container.hide()

        # 5. Bet Cube
        bet_container, lbl_bet = create_cube("ROUND BET 🪙", "bet", "0 $")
        layout.addWidget(bet_container)

        # 6. Bankroll Cube
        # Added a money emoji to the title and changed the value color to neon green
        bank_container, lbl_bank = create_cube("BANKROLL 💵", "bank", "0 $")
        lbl_bank.setStyleSheet("background-color: #111; border: 1px solid #333; border-radius: 5px; color: "
                               "#39ff14; font-size: 26px; font-weight: bold;")  # Neon Green color
        layout.addWidget(bank_container)

        return frame

    def _connect_signals(self):
        self.signals.video_update.connect(self.video_widget.update_frame)
        self.signals.update_timer.connect(self.update_countdown_timer)
        self.signals.update_bet.connect(self._update_bet_display)
        self.signals.update_hand.connect(self._update_hand_display)
        self.signals.game_alert.connect(self._handle_game_alert)
        self.signals.update_strategy.connect(self.update_strategy_display)
        self.signals.update_bankroll.connect(self._update_bankroll_display)
        self.signals.show_decision_prompt.connect(self._toggle_decision_prompt)
        self.signals.start_timer_signal.connect(self.start_decision_timer)
        self.signals.stop_timer_signal.connect(self.stop_decision_timer)
        self.signals.round_over_signal.connect(self.show_end_round_options)
        self.signals.show_shuffle_button.connect(self._toggle_shuffle_button)
        self.signals.financial_result.connect(self._show_financial_result)
        self.signals.update_phase.connect(self._update_phase_display)

    # --- Slots & Logic ---
    def start_calibration_process(self):
        if hasattr(self, 'rois') and self.rois:
            # We already have valid ROIs! Skip calibration and jump straight to the game.
            self.show_setup_dialog()
        else:
            self.control_panel.setCurrentIndex(0)
            self.roi_index = 0

            target = self.required_rois[self.roi_index]
            self.video_widget.set_calibration_mode(True, target)
            self.lbl_instruction.setText(
                f"Step {self.roi_index + 1}/{len(self.required_rois)}\nDraw rectangle for {target}")
            self.btn_confirm_roi.setEnabled(False)

    def _start_game_with_data(self, data: dict):
        """Starts the game using the data provided from the SetupDialog."""
        self.btn_confirm_roi.hide()
        self.p1_name_val = data["p1_name"]
        self.p2_name_val = data["p2_name"]
        self.current_game_mode = data["mode"]

        # ========================================================
        # DYNAMIC UI/UX: Update strategy title and style based on game mode
        # ========================================================
        if self.current_game_mode == "Super Computer":
            # Matrix/Tech styling - Neon cyan color, monospace font for analytical feel
            sc_style = "font-size: 32px; color: #00e5ff; font-family: 'Courier New'; font-weight: bold; " \
                       "letter-spacing: 3px; border: none; margin-bottom: 8px;"
            self.p1_strategy_display.lbl_title.setText("JACK SAYS:")
            self.p1_strategy_display.lbl_title.setStyleSheet(sc_style)
            self.p2_strategy_display.lbl_title.setText("JACK SAYS:")
            self.p2_strategy_display.lbl_title.setStyleSheet(sc_style)

        elif self.current_game_mode == "The Perfect Gambler":
            # Classic Casino VIP styling - Casino gold color, elegant spacing
            pg_style = "font-size: 32px; color: #f4c430; font-weight: bold; letter-spacing: 2px; border: none; " \
                       "margin-bottom: 35px;"
            self.p1_strategy_display.lbl_title.setText("JACK SAYS:")
            self.p1_strategy_display.lbl_title.setStyleSheet(pg_style)
            self.p2_strategy_display.lbl_title.setText("JACK SAYS:")
            self.p2_strategy_display.lbl_title.setStyleSheet(pg_style)

        else:
            # Regular mode - Completely hide the strategy widgets to keep a clean UI
            self.p1_strategy_display.hide()
            self.p2_strategy_display.hide()
        # ========================================================

        self.signals.game_start_request.emit(data)

        self.lbl_p1_name.setText(self.p1_name_val)
        self.lbl_p2_name.setText(self.p2_name_val)
        self.lbl_p1_bank.setText(f"{data['p1_bankroll']:.0f} $")
        self.lbl_p2_bank.setText(f"{data['p2_bankroll']:.0f} $")

        # Update House Rules display with data from setup
        num_decks = data.get("num_decks", 2)

        self.control_panel.hide()
        self.lbl_calib_title.hide()
        self.lbl_instruction.hide()

        self.p1_widget.show()
        self.p2_widget.show()
        self.dealer_frame.show()
        self.lbl_timer.show()
        self.timer_bar.show()

        # Logic for showing the Prediction Cubes ONLY in Super Computer mode
        p1_predict_cont = self.p1_widget.findChild(QWidget, "predict_container_Player_1")
        p2_predict_cont = self.p2_widget.findChild(QWidget, "predict_container_Player_2")

        # Logic for showing the Prediction Cubes ONLY in Super Computer mode
        p1_predict_cont = self.p1_widget.findChild(QWidget, "predict_container_Player_1")
        p2_predict_cont = self.p2_widget.findChild(QWidget, "predict_container_Player_2")

        # Logic for showing the AI Strategy ONLY in Super Computer mode
        if self.current_game_mode == "Super Computer":
            self.dealer_forecast_widget.show()
            self.bet_recommendation_widget.show()
            # if p1_predict_cont: p1_predict_cont.show()
            # if p2_predict_cont: p2_predict_cont.show()

            # Release width restriction so the matrix can expand dynamically
            self.p1_strategy_display.setMaximumWidth(16777215)
            self.p2_strategy_display.setMaximumWidth(16777215)
        else:
            self.dealer_forecast_widget.hide()
            self.bet_recommendation_widget.hide()
            if p1_predict_cont:
                p1_predict_cont.hide()
            if p2_predict_cont:
                p2_predict_cont.hide()

            # Fix the width for Perfect Gambler mode so the center layout
            self.p1_strategy_display.setFixedWidth(550)
            self.p2_strategy_display.setFixedWidth(550)

        # Strategy widgets for individual players always start hidden
        self.p1_strategy_display.hide()
        self.p2_strategy_display.hide()

    def _advance_calibration(self):
        self.roi_index += 1
        if self.roi_index < len(self.required_rois):
            target = self.required_rois[self.roi_index]
            self.video_widget.set_calibration_mode(True, target)
            self.lbl_instruction.setText(
                f"Step {self.roi_index + 1}/{len(self.required_rois)}\nDraw rectangle for {target}")

            self.btn_confirm_roi.setEnabled(False)
            self.btn_confirm_roi.setStyleSheet(
                "background-color: #444; color: #888; font-weight: bold; padding: 12px; border-radius: 6px;")
        else:
            self.video_widget.set_calibration_mode(False)
            self.btn_confirm_roi.setStyleSheet(
                "background-color: #444; color: #888; font-weight: bold; padding: 12px; border-radius: 6px;")
            self.btn_confirm_roi.hide()
            self.signals.calibration_complete.emit(self.rois)
            self.show_setup_dialog()

    @pyqtSlot(str, tuple)
    def _on_roi_captured(self, name, rect):
        self.rois[name] = rect
        self.btn_confirm_roi.setEnabled(True)
        self.lbl_instruction.setText(f"{name} Selected!\nPress Confirm to continue.")
        self.btn_confirm_roi.setStyleSheet(
            "background-color: #00e5ff; color: #000; font-weight: bold; padding: 12px; border-radius: 6px;")

    '''
    @pyqtSlot(str, str)
    def _handle_game_alert(self, entity: str, msg: str):
        """Routes player alerts to their specific status cubes, keeping the UI structure."""
        base_cube_style = "background-color: #1c1c1c; border: none; border-radius: 8px; font-size: 20px; " \
                          "font-weight: bold; "

        if msg == "":
            if (entity == self.p2_name_val) or (entity == "Player 2"):
                if self.lbl_p2_status:
                    self.lbl_p2_status.setText("--")
                    self.lbl_p2_status.setStyleSheet(base_cube_style + "color: #444;")
            elif (entity == self.p1_name_val) or (entity == "Player 1"):
                if self.lbl_p1_status:
                    self.lbl_p1_status.setText("--")
                    self.lbl_p1_status.setStyleSheet(base_cube_style + "color: #444;")
            elif entity == "SYSTEM" or entity == "Dealer":
                self.alert_box.hide()
                self.alert_box.setText("")
            else:
                self.alert_box.hide()
                self.alert_box.setText("")
                if self.lbl_p1_status:
                    self.lbl_p1_status.setText("--")
                    self.lbl_p1_status.setStyleSheet(base_cube_style + "color: #444;")
                if self.lbl_p2_status:
                    self.lbl_p2_status.setText("--")
                    self.lbl_p2_status.setStyleSheet(base_cube_style + "color: #444;")
            return

        target_label = None
        if (entity == self.p2_name_val) or (entity == "Player 2"):
            target_label = self.lbl_p2_status
        elif (entity == self.p1_name_val) or (entity == "Player 1"):
            target_label = self.lbl_p1_status

        if target_label:
            target_label.setText(msg)
            msg_upper = msg.upper()

            if "BUSTED" in msg_upper or "LOSE" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: #ff3333;")
                self.play_sound("lose")
            elif "BLACKJACK" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: #ffd700;")
                self.play_sound("blackjack")
            elif "WIN" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: #ffd700;")
                self.play_sound("win")
            elif "PUSH" in msg_upper or "TIE" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: white;")
                self.play_sound("push")
            elif "HIT" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: white;")
                # Delay sound by 2 seconds (2000 ms) when player HITS
                QTimer.singleShot(2000, lambda: self.play_sound("new_card"))
            elif "SITTING" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: #888;")
            elif "ALL IN" in msg_upper:
                # Make the text pop with red color and bold weight
                target_label.setStyleSheet(base_cube_style + "color: #ff4444; font-size: 22px; font-weight: bold;")
                # Play the hype sound!
                self.play_sound("all in")
                QTimer.singleShot(2500, lambda: self.stop_sound("all in"))
            else:
                target_label.setStyleSheet(base_cube_style + "color: white;")
        else:
            full_msg = f"{entity}\n{msg}" if entity and entity != "SYSTEM" else msg
            msg_upper = msg.upper()

            if "BUSTED" in msg_upper or "LOSE" in msg_upper:
                style = "background-color: rgba(211, 47, 47, 0.9); border: 2px solid #ff5252; color: white;"
                self.play_sound("lose")
            elif "BLACKJACK" in msg_upper:
                style = "background-color: rgba(255, 215, 0, 0.9); border: 2px solid #fff; color: black;"
                self.play_sound("blackjack")
            elif "WIN" in msg_upper:
                style = "background-color: rgba(255, 215, 0, 0.9); border: 2px solid #fff; color: black;"
                self.play_sound("win")
            elif "DEALING DEALER CARD" in msg_upper:
                style = "background-color: rgba(2, 136, 209, 0.9); color: white;"
                # Update the label so update_countdown_timer knows to treat this as a 'regular' phase
                self._decision_label = "DEALING DEALER CARD"
                # Keep the 2-second delay for the new card sound as requested earlier
                QTimer.singleShot(2000, lambda: self.play_sound("new_card"))
            else:
                style = "background-color: rgba(2, 136, 209, 0.9); color: white;"

            self.alert_box.setText(full_msg)
            self.alert_box.setStyleSheet(
                f"{style} border-radius: 10px; padding: 20px; font-size: 36px; font-weight: bold;")
            self.alert_box.show()

        if entity in ["SYSTEM", "Dealer", ""]:
            if msg == "":
                return

            self.lbl_timer.setText(msg.upper())
            self.lbl_timer.setStyleSheet("font-size: 22px; color: #ffeb3b; font-weight: bold;")
            self.lbl_timer.show()
    '''

    @pyqtSlot(str, str)
    def _handle_game_alert(self, entity: str, msg: str):
        """Routes player alerts to their specific status cubes, and system/dealer alerts to the bottom box."""
        base_cube_style = "background-color: #1c1c1c; border: none; border-radius: 8px; font-size: 28px; font-weight: bold; "

        if msg == "":
            if (entity == self.p2_name_val) or (entity == "Player 2"):
                if self.lbl_p2_status:
                    self.lbl_p2_status.setText("--")
                    self.lbl_p2_status.setStyleSheet(base_cube_style + "color: #444;")
            elif (entity == self.p1_name_val) or (entity == "Player 1"):
                if self.lbl_p1_status:
                    self.lbl_p1_status.setText("--")
                    self.lbl_p1_status.setStyleSheet(base_cube_style + "color: #444;")
            elif entity in ["SYSTEM", "Dealer", ""]:
                self.lbl_timer.hide()
                self.lbl_timer.setText("")
            return

        target_label = None
        if (entity == self.p2_name_val) or (entity == "Player 2"):
            target_label = self.lbl_p2_status
        elif (entity == self.p1_name_val) or (entity == "Player 1"):
            target_label = self.lbl_p1_status

        if target_label:
            target_label.setText(msg)
            msg_upper = msg.upper()

            # Cutting off any previous long sound (like a long win sound) before playing a new result
            if any(x in msg_upper for x in ["BUSTED", "LOSE", "BLACKJACK", "WIN", "PUSH", "TIE"]):
                pygame.mixer.stop()

            if "BUSTED" in msg_upper or "LOSE" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: #ff3333;")
                self.play_sound("lose")
            elif "BLACKJACK" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: #ffd700;")
                self.play_sound("blackjack")
            elif "WIN" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: #ffd700;")
                self.play_sound("win")
            elif "PUSH" in msg_upper or "TIE" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: white;")
                self.play_sound("push")
            elif "HIT" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: white;")
                QTimer.singleShot(2000, lambda: self.play_sound("new_card"))
            elif "SITTING" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: #888;")
            elif "ALL IN" in msg_upper:
                target_label.setStyleSheet(base_cube_style + "color: #ff4444; font-size: 28px; font-weight: bold;")
                self.play_sound("all in")
                QTimer.singleShot(2500, lambda: self.stop_sound("all in"))
            else:
                target_label.setStyleSheet(base_cube_style + "color: white;")

        elif entity in ["SYSTEM", "Dealer"]:
            msg_upper = msg.upper()
            self.lbl_timer.setText(msg_upper)
            if "BUSTED" in msg_upper or "LOSE" in msg_upper:
                self.lbl_timer.setStyleSheet(
                    self.base_bottom_cube + "font-size: 28px; color: #ff5252; font-weight: bold;")
                self.play_sound("lose")
            elif "BLACKJACK" in msg_upper or "WIN" in msg_upper:
                self.lbl_timer.setStyleSheet(
                    self.base_bottom_cube + "font-size: 28px; color: #ffd700; font-weight: bold;")
            elif "DEALING DEALER CARD" in msg_upper:
                self.lbl_timer.setStyleSheet(
                    self.base_bottom_cube + "font-size: 28px; color: #00e5ff; font-weight: bold;")
                self._decision_label = "Dealing dealer card"
                QTimer.singleShot(2000, lambda: self.play_sound("new_card"))
            else:
                self.lbl_timer.setStyleSheet(self.base_bottom_cube + "font-size: 28px; color: #aaa; font-weight: bold;")
            self.lbl_timer.show()

    @pyqtSlot(str, int)
    def _update_bet_display(self, player_name: str, amount: int):
        """Updates the bet cube. Gray if 0, Neon Green if > 0."""
        is_p2 = (player_name == self.p2_name_val) or (player_name == "Player 2")
        target = self.lbl_p2_bet if is_p2 else self.lbl_p1_bet

        base_style = "background-color: #1c1c1c; border: none; border-radius: 8px; font-size: 26px; font-weight: bold;"

        if amount == 0:
            target.setText("0 $")
            target.setStyleSheet(base_style + "color: #888;")
        else:
            target.setText(f"{amount} $")
            target.setStyleSheet(base_style + "color: #4caf50;")  # classic casino green

    @pyqtSlot(str, str)
    def _update_hand_display(self, entity: str, display_text: str):
        """Updates the hand cube dynamically while preserving the background."""
        if "Dealer" in entity:
            target = self.lbl_dealer_score
            if display_text == "0":
                target.setText("--")
                target.setStyleSheet("font-size: 70px; font-weight: bold; color: #444;")
            else:
                target.setText(display_text)

                # --- DYNAMIC DEALER STYLING ---
                if display_text == "BLACKJACK":
                    # Gold color for Blackjack (#ffd700)
                    target.setStyleSheet("font-size: 34px; font-weight: bold; color: #ffd700;")
                else:
                    # Electric Orange for regular numbers (#ff9800)
                    target.setStyleSheet("font-size: 70px; font-weight: bold; color: #ff9800;")
            return

        is_p2 = (entity == self.p2_name_val) or (entity == "Player 2")
        target = self.lbl_p2_hand if is_p2 else self.lbl_p1_hand
        accent = "#bd00ff" if is_p2 else "#00e5ff"

        base_style = "background-color: #1c1c1c; border: none; border-radius: 8px; font-size: 70px; font-weight: bold;"

        if display_text == "0":
            target.setText("--")
            target.setStyleSheet(base_style + "color: #444;")
        else:
            target.setText(display_text)
            target.setStyleSheet(base_style + f"color: {accent};")

    @pyqtSlot(str, str, object)
    def update_strategy_display(self, player_name: str, action: str, stats: Optional[Dict[str, Any]]):
        """Processes and routes strategy data to the correct P1/P2 widget for ALL game modes."""

        # 1. Clean slate: Always hide before re-evaluating turn
        self.p1_strategy_display.hide()
        self.p2_strategy_display.hide()

        if getattr(self, 'current_game_mode', '') == "Regular" or not action or action == "CLEAR":
            return

        if player_name == "Dealer":
            if stats and "dealer_probs" in stats:
                raw_d_probs = stats["dealer_probs"]
                d_probs_pct = {k: v * 100 for k, v in raw_d_probs.items()}
                self.dealer_forecast_widget.update_forecast(d_probs_pct)
            return

        # 2. Routing: Identify if the data belongs to Player 1 or Player 2
        is_p1 = (player_name == self.p1_name_val) or ("1" in player_name)
        active_strategy_widget = self.p1_strategy_display if is_p1 else self.p2_strategy_display

        title_text = "JACK SAYS:" if getattr(self, 'current_game_mode',
                                             '') == "Super Computer" else "JACK SAYS:"
        title_color = "#00e5ff" if getattr(self, 'current_game_mode', '') == "Super Computer" else "#f4c430"
        active_strategy_widget.lbl_title.setText(title_text)
        active_strategy_widget.lbl_title.setStyleSheet(
            f"font-size: 32px; color: {title_color}; font-weight: bold; letter-spacing: 2px; border: none; "
            f"margin-bottom: 8px;")

        # --- Case A: Pre-Round Evaluation (Common for SC mode) ---
        if player_name == "PRE_ROUND" and stats:
            self.bet_recommendation_widget.show()
            raw_data = stats.get("stats", {})
            p_stats = raw_data.get("player_stats", {})
            bet_rec = raw_data.get("bet_recommendation", "")

            # Fetch the EV so the speedometer needle can move!
            ev = raw_data.get("ev", 0.0)

            # Pass the EV to the widget
            self.bet_recommendation_widget.update_data(bet_rec, p_stats, ev)

            # Standardize Dealer Forecast to Percentages (0-100)
            raw_d_probs = raw_data.get("dealer_probs", {})
            self.dealer_forecast_widget.update_forecast(raw_d_probs)
            return

        # 3. Activation: Show the relevant side-widget (Works for BOTH modes now!)
        active_strategy_widget.show()
        action_upper = action.upper()

        # --- Case B: Super Computer Mode (Active Decision Turn) ---
        if stats and "all_actions" in stats:
            self.bet_recommendation_widget.hide()
            active_strategy_widget.update_recommendation(action_upper, stats)

            # During the active decision phase, ALWAYS keep the player's personal prediction cube empty!
            self._update_player_prediction_label(player_name, None)

            # Update Dealer Forecast
            raw_d_probs = stats.get("dealer_probs", {})
            d_probs_pct = {k: v * 100 for k, v in raw_d_probs.items()}
            self.dealer_forecast_widget.update_forecast(d_probs_pct)

        # --- Case C: Terminal Statuses (End of Turn) OR Perfect Gambler Mode ---
        else:
            active_strategy_widget.update_recommendation(action_upper, stats)

            # If this is the end of a turn, 'stats' contains {win, tie, loss} and will populate the side cube.
            # If it's just a Perfect Gambler active turn, 'stats' is None and the cube stays hidden.
            self._update_player_prediction_label(player_name, stats)

            # Hide the top strategy widget completely if the turn is over
            if action_upper in ["DONE", "BUSTED", "SURRENDERED", "BLACKJACK"]:
                active_strategy_widget.hide()

    def _update_player_prediction_label(self, player_name, flat_stats):
        """Updates the prediction cube. Hides the entire cube if no data is available."""
        target_label = self.lbl_p2_prediction if "2" in player_name or player_name == self.p2_name_val \
            else self.lbl_p1_prediction

        # Get the container (the parent widget that holds both the title and the label)
        container = target_label.parentWidget()

        base_style = "background-color: #111; border: 1px solid #333; border-radius: 5px; font-size: 22px; " \
                     "font-weight: bold; "

        if flat_stats and isinstance(flat_stats, dict) and "win" in flat_stats:
            w = int(flat_stats.get("win", 0) * 100)
            t = int(flat_stats.get("tie", 0) * 100)
            l = int(flat_stats.get("loss", 0) * 100)

            '''
            target_label.setText(f"WIN:{w}% TIE:{t}% LOSS:{l}%")

            # Dynamic coloring based on win probability
            color = "#00e5ff" if w >= 50 else "#ff5555" if w < 40 else "#ccc"
            target_label.setStyleSheet(base_style + f"color: {color};")
            '''

            formatted_text = (
                f'<span style="color: #39ff14;">WIN:{w}%</span>&nbsp;&nbsp;'
                f'<span style="color: #aaaaaa;">TIE:{t}%</span>&nbsp;&nbsp;'
                f'<span style="color: #ff5555;">LOSS:{l}%</span>'
            )
            target_label.setText(formatted_text)
            target_label.setStyleSheet(base_style)

            # Show the entire cube with its title
            container.show()
        else:
            # Hide the entire cube completely (the UI won't jump because of RetainSizeWhenHidden)
            container.hide()

    @pyqtSlot(bool)
    def _toggle_decision_prompt(self, show):
        if show:
            self.lbl_timer.setText("Waiting for decision...")
            self.lbl_timer.setStyleSheet("color: #00e5ff; font-size: 22px; font-weight: bold;")
        else:
            self.lbl_timer.setText("")

    @pyqtSlot(str, int)
    def _update_bankroll_display(self, player, amount):
        """Updates bankroll with a rolling numbers animation (1 second) and sound."""

        if not hasattr(self, 'bankroll_timers'):
            self.bankroll_timers = {}

        is_p2 = (player == self.p2_name_val) or (player == "Player 2")
        target_label = self.lbl_p2_bank if is_p2 else self.lbl_p1_bank

        current_text = target_label.text().replace(" $", "")
        try:
            current_amount = int(current_text)
        except ValueError:
            current_amount = 0

        if current_amount == amount:
            return

        is_win = amount > current_amount
        flash_color = "#39ff14" if is_win else "#ff5555"
        base_color = "#39ff14"

        target_label.setStyleSheet(f"background-color: #111; border: 2px solid {flash_color}; "
                                   f"border-radius: 5px; color: {flash_color}; font-size: 26px; font-weight: bold;")

        # Play the falling money sound if the balance is decreasing (betting)
        if not is_win:
            self.play_sound("money_drop")

        # Configuration for 1 second animation: 40 steps * 25ms = 1000ms
        steps = 40
        step_duration = 25
        diff = amount - current_amount
        step_val = diff / steps

        if player in self.bankroll_timers:
            self.bankroll_timers[player].stop()

        timer = QTimer(self)
        self.bankroll_timers[player] = timer

        anim_data = {
            "current": float(current_amount),
            "target": amount,
            "step_val": step_val,
            "steps_left": steps
        }

        def animate():
            anim_data["steps_left"] -= 1
            anim_data["current"] += anim_data["step_val"]

            if anim_data["steps_left"] <= 0:
                target_label.setText(f"{anim_data['target']} $")
                target_label.setStyleSheet(f"background-color: #111; border: 1px solid #333; "
                                           f"border-radius: 5px; color: {base_color}; font-size: 26px; font-weight: bold;")
                self.stop_sound("money_drop")  # Stop the sound when finished
                timer.stop()
            else:
                target_label.setText(f"{int(anim_data['current'])} $")

        timer.timeout.connect(animate)
        timer.start(step_duration)

    @pyqtSlot(str, int, int)
    def _show_financial_result(self, player_name: str, payout: int, bet: int):
        is_p2 = (player_name == self.p2_name_val) or (player_name == "Player 2")
        target_widget = self.p2_result_widget if is_p2 else self.p1_result_widget
        target_widget.show_result(payout, bet)

    # --- Timer Slots ---
    @pyqtSlot(int, str)
    def start_decision_timer(self, seconds: int, label: str):
        """Starts a countdown with the given duration and phase label."""
        self._decision_total = seconds
        self._decision_remaining = seconds
        self._decision_label = label

        # Set up the label and progress bar
        self.lbl_timer.setText(label)
        self.lbl_timer.setStyleSheet(self.base_bottom_cube + "font-size: 28px; color: #aaa; font-weight: bold;")
        self.timer_bar.setMaximum(seconds)
        self.timer_bar.setValue(seconds)
        self.lbl_timer.show()
        self.timer_bar.show()

        # Reset color to yellow
        self.timer_bar.setStyleSheet("""
            QProgressBar { background-color: #333; border: none; border-radius: 4px; }
            QProgressBar::chunk { background-color: #ffeb3b; border-radius: 4px; }
        """)

        # Convert label to uppercase for safe checking
        label_upper = label.upper()

        # Start looping the dealing sound if it's the deal phase
        if label_upper == "DEALING INITIAL CARDS...":
            if self.sounds.get("deal"):
                self.sounds["deal"].play(loops=-1)

        # Start looping the chips sound if it's the betting phase
        elif label_upper == "PLACE YOUR BETS":
            if self.sounds.get("chips"):
                self.sounds["chips"].play(loops=-1)

    @pyqtSlot()
    def stop_decision_timer(self):
        """Immediately stops the timer and shows the closed state."""
        self.stop_sound("deal")
        self.stop_sound("chips")
        self.update_countdown_timer(0)
    '''
    @pyqtSlot(int)
    def update_countdown_timer(self, seconds_left: int) -> None:
        """
        Updates the UI timer dynamically based on the current phase.
        Exempts the 'DEAL INITIAL CARDS' phase from turning red and beeping.
        """
        current_label = getattr(self, '_decision_label', 'WAITING')

        # Convert to uppercase to ensure "BETS" is always found
        current_label_upper = current_label.upper()

        if seconds_left > 0:
            # Update the progress bar visually
            self.timer_bar.setValue(seconds_left)

            # Check if we are in the last 5 seconds and NOT in any dealing phase
            if "DEAL" not in current_label_upper and seconds_left <= 5:
                # 5 seconds or less -> RED BAR + AUDIO TICK
                if current_label_upper == "PLACE YOUR BETS" and seconds_left == 5:
                    self.stop_sound("chips")

                self.play_sound("tick")

                # Creates a flashing effect by alternating bright and dark red every second
                flash_color = "#ff5555" if seconds_left % 2 != 0 else "#880000"

                # Change bar and text color to the flashing red warning
                self.timer_bar.setStyleSheet(f"""
                                    QProgressBar {{ background-color: #333; border: none; border-radius: 4px; }}
                                    QProgressBar::chunk {{ background-color: {flash_color}; border-radius: 4px; }}
                                """)
                self.lbl_timer.setStyleSheet(self.base_bottom_cube + "font-size: 28px; color: #ff5555; "
                                                                     "font-weight: bold;")
        else:
            # Time's up - Stop looping sounds and display final text
            self.stop_sound("deal")
            self.stop_sound("chips")

            self.timer_bar.setValue(0)

            if "BETS" in current_label_upper:
                final_text = "No More Bets!"
                self.play_sound("no_more_bets")
            elif "DECISION" in current_label_upper:
                final_text = "Time's up"
            elif "DEAL" in current_label_upper:
                final_text = "Reading Cards..."
            else:
                final_text = f"{current_label} done"

            self.lbl_timer.setText(final_text)
            self.lbl_timer.setStyleSheet(self.base_bottom_cube + "font-size: 28px; color: #ff5555; font-weight: bold;")
    '''

    @pyqtSlot(int)
    def update_countdown_timer(self, seconds_left: int) -> None:
        """
        Updates the UI timer elements using positive logic.
        Only 'BETS' and 'DECISION' phases get the flashing red warning style.
        """
        # Safely retrieve the current label and convert to uppercase
        current_label = getattr(self, '_decision_label', 'WAITING')
        current_label_upper = current_label.upper()

        if seconds_left > 0:
            # Update progress bar value
            self.timer_bar.setValue(seconds_left)

            # --- WARNING MODE (Positive Logic) ---
            # Triggered only for Betting or Decision phases in the final 5 seconds
            if ("BETS" in current_label_upper or "DECISION" in current_label_upper) and seconds_left <= 5:
                self.play_sound("tick")

                # Dynamic flashing effect (bright red vs dark red)
                flash_color = "#ff5555" if seconds_left % 2 != 0 else "#880000"

                # Apply the detailed 'High-End' style you liked for the progress bar
                self.timer_bar.setStyleSheet(f"""
                        QProgressBar {{ background-color: #333; border: none; border-radius: 4px; }}
                        QProgressBar::chunk {{ background-color: {flash_color}; border-radius: 4px; }}
                    """)

                # Update label text to red
                self.lbl_timer.setStyleSheet(
                    self.base_bottom_cube + "font-size: 28px; color: #ff5555; font-weight: bold;")

            # --- NORMAL/SILENT MODE ---
            # Used for all other states (like Dealing) or when time is > 5s
            else:
                # Standard yellow bar with the same detailed structure
                self.timer_bar.setStyleSheet("""
                        QProgressBar { background-color: #333; border: none; border-radius: 4px; }
                        QProgressBar::chunk { background-color: #ffeb3b; border-radius: 4px; }
                    """)

                # Standard gray label text
                self.lbl_timer.setStyleSheet(self.base_bottom_cube + "font-size: 28px; color: #aaa; font-weight: bold;")

        else:
            # --- TIMER EXPIRED (0 Seconds) ---
            self.stop_sound("deal")
            self.stop_sound("chips")
            self.timer_bar.setValue(0)

            # Define final text and color based on the phase
            if "BETS" in current_label_upper:
                final_text = "No More Bets!"
                final_color = "#ff5555"  # Alarming Red
                self.play_sound("no_more_bets")
            elif "DEAL" in current_label_upper:
                final_text = "WAITING FOR CARD..."
                final_color = "#ffeb3b"  # Neutral Yellow
            elif "DECISION" in current_label_upper:
                final_text = "Time's up"
                final_color = "#ff5555"  # Alarming Red
            else:
                final_text = f"{current_label} DONE"
                final_color = "#aaa"

            # Apply final style
            self.lbl_timer.setText(final_text)
            self.lbl_timer.setStyleSheet(
                self.base_bottom_cube + f"font-size: 28px; color: {final_color}; font-weight: bold;")

    # --- End-of-Round Slots ---
    @pyqtSlot()
    def show_end_round_options(self):
        """Shows the New Round / End Game buttons when the round is fully over."""
        self.btn_container.show()

    def _on_new_round_btn_pressed(self):
        """Hides buttons, shows a status message, and broadcasts new-round intent."""
        self.btn_container.hide()
        self.lbl_p1_hand.setText("--")
        self.lbl_p2_hand.setText("--")
        self.lbl_dealer_score.setText("--")
        self.lbl_p1_prediction.setText("--")
        self.lbl_p2_prediction.setText("--")
        self.lbl_p1_status.setText("--")
        self.lbl_p2_status.setText("--")
        self.p1_strategy_display.hide()
        self.p2_strategy_display.hide()
        self.lbl_p1_prediction.setText("--")
        self.lbl_p2_prediction.setText("--")

        # Ensure the style goes back to gray
        self._update_player_prediction_label(self.p1_name_val, None)
        self._update_player_prediction_label(self.p2_name_val, None)

        self.signals.game_alert.emit("SYSTEM", "Starting New Round...")
        self.signals.new_round_clicked.emit()

        self.p1_result_widget.hide()
        self.p2_result_widget.hide()

    def _on_new_game_btn_pressed(self):
        """Acts as 'Start New Game'. Keeps ROIs, but resets players and mode."""
        self.btn_container.hide()
        self.lbl_p1_hand.setText("--")
        self.lbl_p2_hand.setText("--")
        self.lbl_dealer_score.setText("--")
        self.lbl_p1_prediction.setText("--")
        self.lbl_p2_prediction.setText("--")
        self.lbl_p1_status.setText("")
        self.lbl_p2_status.setText("")
        self.p1_strategy_display.hide()
        self.p2_strategy_display.hide()

        # Ensure the style goes back to gray
        self._update_player_prediction_label(self.p1_name_val, None)
        self._update_player_prediction_label(self.p2_name_val, None)

        self.signals.new_game_clicked.emit()

        self.p1_widget.hide()
        self.p2_widget.hide()
        self.dealer_frame.hide()
        self.lbl_timer.hide()
        if hasattr(self, 'timer_bar'):
            self.timer_bar.hide()  # Hide the progress bar

        self.show_setup_dialog()

        self.p1_result_widget.hide()
        self.p2_result_widget.hide()

    @pyqtSlot(bool)
    def _toggle_shuffle_button(self, show: bool):
        """Shows or hides the manual shuffle completion button and plays audio."""
        if show:
            self.btn_container.show()
            self.btn_new_round.hide()
            self.btn_new_game.hide()
            self.btn_shuffle_done.show()
            # Start looping the shuffle sound
            if self.sounds.get("shuffle"):
                self.sounds["shuffle"].play(loops=-1)
        else:
            self.btn_shuffle_done.hide()
            self.btn_new_round.show()
            self.btn_new_game.show()
            self.btn_container.hide()
            self.stop_sound("shuffle")

    def _on_shuffle_done_pressed(self):
        """Broadcasts that the physical dealer finished shuffling."""
        self.stop_sound("shuffle")  # Stop immediately for snappy UX
        self.signals.shuffle_complete.emit()

    def _on_shuffle_done_pressed(self):
        """Broadcasts that the physical dealer finished shuffling."""
        self.signals.shuffle_complete.emit()

    def keyPressEvent(self, event):
        """
        Global Keyboard Shortcuts for emergency resets and navigation.
        Ensures the game remains recoverable even if physical triggers fail.
        """
        # 1. ESC: Standard Full screen/Close logic
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()

        # 2. SPACE: ROI Confirmation (during calibration)
        elif event.key() == Qt.Key_Space:
            if hasattr(self,
                       'btn_confirm_roi') and self.btn_confirm_roi.isVisible() and self.btn_confirm_roi.isEnabled():
                self.btn_confirm_roi.click()

        # ========================================================
        # EMERGENCY RESET KEYS (Force State Transitions)
        # ========================================================

        # 3. 'R' Key: Force New Round
        # Triggers the same logic as the "New Round" button unconditionally!
        elif event.key() == Qt.Key_R:
            print("[SYSTEM] Keyboard Override: Forcing New Round (Key R)")
            self._on_new_round_btn_pressed()

        # 4. 'N' Key: Force New Game
        # Triggers the same logic as the "New Game" button unconditionally!
        elif event.key() == Qt.Key_N:
            print("[SYSTEM] Keyboard Override: Forcing New Game Setup (Key N)")
            self._on_new_game_btn_pressed()

    @pyqtSlot(str)
    def _update_phase_display(self, phase_text):
        """Updating the top cube - game stage"""
        if not phase_text:
            self.alert_box.hide()
            return

        self.alert_box.setText(phase_text.upper())
        style = """
                    background-color: #1a1a2e; 
                    border: 3px solid #ccff00; 
                    color: #ccff00; 
                    border-radius: 10px; 
                    padding: 10px; 
                    font-size: 36px; 
                    font-weight: bold;
                """
        self.alert_box.setStyleSheet(style)
        self.alert_box.show()
