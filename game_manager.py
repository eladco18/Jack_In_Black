import time
import math
from typing import List, Dict, Optional, Any
from enum import Enum
from vision_engine import stabilize_detection


class PlayerStatus(Enum):
    PLAYING = "PLAYING"
    DONE = "DONE"
    BUSTED = "BUSTED"
    SURRENDERED = "SURRENDERED"
    BLACKJACK = "BLACKJACK"
    PUSH = "PUSH"
    WIN = "WIN"
    LOSE = "LOSE"


class GameManager:
    """
    The central State Machine (Conductor) that controls the flow of the Blackjack game.
    Orchestrates the computer vision engine, the game logic, and updates the UI.
    Strictly adheres to defined game states.
    """

    def __init__(self) -> None:
        """
        Initializes the GameManager with default states and empty game variables.
        """
        # State mapping constants
        self.STATE_CALIBRATION: int = 0
        self.STATE_INIT_GAME: int = 1
        self.STATE_BETTING_DEALING: int = 2
        self.STATE_PLAYER_TURN: int = 3
        self.STATE_DEALER_TURN: int = 4
        self.STATE_COMPARISON: int = 5
        self.STATE_BANKROLL_UPDATE: int = 6
        self.STATE_WAIT_FOR_NEXT_ROUND: int = 7
        self.STATE_WAIT_FOR_SHUFFLE: int = 8

        # Current active state
        self.current_state: int = self.STATE_CALIBRATION

        # Game Data Structures
        self.players: List[Dict[str, Any]] = []
        self.dealer_hand: List[int] = []
        self.dealer_locked_cards: List[List[int]] = []
        self.dealer_effective_sum: int = 0
        self.dealer_has_ace: bool = False
        self.game_mode: str = "The Perfect Gambler"  # Options: "The Perfect Gambler", "Super Computer"

        # Pointers to other modules (to be injected during main setup)
        self.vision_engine: Optional[Any] = None
        self.game_logic: Optional[Any] = None
        self.ui_manager: Optional[Any] = None
        self.camera_thread: Optional[Any] = None
        self.signals: Optional[Any] = None
        self.abort_flag: bool = False

    def inject_dependencies(self, vision_engine: Any, game_logic: Any, ui_manager: Any, camera_thread: Any = None,
                            signals: Any = None) -> None:
        """
        Injects instantiated external modules into the GameManager.
        This allows the centralized state machine to interact with the computer vision
        system, mathematical logic, UI components, and the event bus without tightly
        coupling their instantiations.

        Args:
            vision_engine (Any): The module handling image processing and object detection.
            game_logic (Any): The module calculating sums, managing bankrolls, and wrapping the strategy engine.
            ui_manager (Any): The module handling the graphical user interface updates.
            camera_thread (Any, optional): The thread continuously pulling frames from the camera.
            signals (Any, optional): The PyQt Event Bus used to emit signals to the UI.
        """
        self.vision_engine = vision_engine
        self.game_logic = game_logic
        self.ui_manager = ui_manager
        self.camera_thread = camera_thread
        self.signals = signals

    def run_game_loop(self) -> None:
        """
        Main continuous game loop. Calls the appropriate method based on the current state.
        Runs continuously in a background thread.
        """
        while True:
            try:
                self.abort_flag = False

                if self.current_state == self.STATE_CALIBRATION:
                    self.state_0_calibration()
                elif self.current_state == self.STATE_INIT_GAME:
                    self.state_1_init_game()
                elif self.current_state == self.STATE_BETTING_DEALING:
                    self.state_2_betting_and_dealing()
                elif self.current_state == self.STATE_PLAYER_TURN:
                    self.state_3_player_turn()
                elif self.current_state == self.STATE_DEALER_TURN:
                    self.state_4_dealer_turn()
                elif self.current_state == self.STATE_COMPARISON:
                    self.state_5_comparison()
                elif self.current_state == self.STATE_BANKROLL_UPDATE:
                    self.state_6_bankroll_update()
                elif self.current_state == self.STATE_WAIT_FOR_NEXT_ROUND:
                    self.state_7_wait_for_next_round()
                elif self.current_state == self.STATE_WAIT_FOR_SHUFFLE:
                    pass  # Idle state, wait for UI button

                # Prevent CPU overload and allow the UI thread to breathe
                time.sleep(0.1)

            except RuntimeError:
                # Silently catch the teardown exception when the GUI is closed by the user
                break
            except Exception as e:
                print(f"[GAME LOOP] Non-fatal error: {e}")
                time.sleep(1)

    def state_0_calibration(self) -> None:
        """
        Pre-game setup: Pauses the game loop until ROI (Region of Interest)
        calibration is completed via the UI and Vision Engine.
        """
        # Implement check to see if calibration is done, then:
        # self.current_state = self.STATE_INIT_GAME
        pass

    def state_1_init_game(self) -> None:
        """
        Step 1: Game Start.
        Triggered externally via UI. Sets game mode, player names, and initial bankrolls.
        """
        # Populate self.players based on UI inputs
        # Reset card inventory via self.game_logic._initialize_decks()
        # self.current_state = self.STATE_BETTING_DEALING
        pass

    def state_2_betting_and_dealing(self) -> None:
        """
        Step 2: Start Round.
        Checks for reshuffle, resets round data, handles the betting phase,
        deducts bets from bankrolls, and deals the initial cards.
        """
        required_rois = ["player_1_chips", "player_2_chips", "player_1_cards", "player_2_cards", "dealer_cards"]
        for roi in required_rois:
            if not self.vision_engine.rois.get(roi):
                self.signals.game_alert.emit("SYSTEM",
                                             f"CRITICAL: Missing ROI Calibration for {roi}. Restarting setup.")
                time.sleep(4)
                self.current_state = self.STATE_CALIBRATION
                return

        # -1. RESHUFFLE CHECK:
        #  Check if we reached the dynamically generated Cut Card threshold
        if self.game_logic.needs_reshuffle():
            # Calculate remaining cards
            cards_left = sum(self.game_logic.card_inventory.values())

            # Set the video message
            msg = f"{cards_left} CARDS LEFT\nPLEASE RESHUFFLE THE SHOE"
            self.vision_engine.set_system_overlay(msg)
            self.signals.update_phase.emit("SHUFFLING SHOE")
            self.signals.show_shuffle_button.emit(True)
            self.current_state = self.STATE_WAIT_FOR_SHUFFLE
            return

        # 0. RESET ROUND DATA
        for player in self.players:
            player["current_bet"] = 0
            player["hand"] = []
            player["locked_cards"] = []
            player["effective_sum"] = 0
            player["has_ace"] = False
            player["status"] = PlayerStatus.PLAYING
            player["has_doubled"] = False

        self.dealer_hand = []
        self.dealer_locked_cards = []
        self.dealer_effective_sum = 0
        self.dealer_has_ace = False

        # Update UI to clear previous round displays
        self.signals.game_alert.emit("", "")
        self.signals.update_bet.emit(self.players[0]["name"], 0)
        self.signals.update_bet.emit(self.players[1]["name"], 0)
        self.signals.update_hand.emit("Dealer", "0")
        self.signals.update_hand.emit(self.players[0]["name"], "0")
        self.signals.update_hand.emit(self.players[1]["name"], "0")

        # -------------
        # BETTING PHASE
        # -------------
        self.signals.update_phase.emit("BETTING PHASE")
        self.signals.game_alert.emit("SYSTEM", "Place Your Bets!")
        if self.game_mode == "Super Computer":
            pre_round_data = self.game_logic.get_pre_round_evaluation()
            self.signals.update_strategy.emit(
                "PRE_ROUND",
                pre_round_data["bet_recommendation"],
                {"ev": pre_round_data["ev"], "stats": pre_round_data}
            )
        self.signals.start_timer_signal.emit(15, "Place Your Bets")
        for remaining_seconds in range(15, -1, -1):
            self.signals.update_timer.emit(remaining_seconds)
            time.sleep(1)
        self.signals.stop_timer_signal.emit()
        time.sleep(2)
        self.signals.game_alert.emit("SYSTEM", "Reading Chips...")
        time.sleep(3)

        # 1. Fetch a sequence of recent frames from the camera's buffer
        recent_frames = self.camera_thread.get_recent_frames(num_frames=15)

        # 2. Lock Player 1's bet using the stabilization wrapper
        p1_x1, p1_y1, p1_x2, p1_y2 = self.vision_engine.rois["player_1_chips"]
        p1_cropped = [f[p1_y1:p1_y2, p1_x1:p1_x2] for f in recent_frames]
        p1_locked_bet = stabilize_detection(
            frames_list=p1_cropped,
            detection_func=self.vision_engine.analyze_player_bet,
            player_name_vision=1
        )
        if p1_locked_bet is None:
            p1_locked_bet = 0

        # 3. Lock Player 2's bet using the exact same wrapper
        p2_x1, p2_y1, p2_x2, p2_y2 = self.vision_engine.rois["player_2_chips"]
        p2_cropped = [f[p2_y1:p2_y2, p2_x1:p2_x2] for f in recent_frames]
        p2_locked_bet = stabilize_detection(
            frames_list=p2_cropped,
            detection_func=self.vision_engine.analyze_player_bet,
            player_name_vision=2
        )
        if p2_locked_bet is None:
            p2_locked_bet = 0

        # 4. Update the internal game state with the locked bets
        anyone_all_in = False
        needs_rebet = False
        players_to_clear_video = []

        # --- PASS 1: VALIDATION ONLY (Do not deduct money yet) ---
        for i, locked_bet in enumerate([p1_locked_bet, p2_locked_bet]):
            player = self.players[i]
            if locked_bet > player["bankroll"]:
                self.signals.game_alert.emit(player["name"], "OVERBET! ⚠️")
                self.vision_engine.set_player_overlay(i + 1, "Overbet detected!\nPlease adjust")
                needs_rebet = True

        # If anyone overbet, abort the entire transaction and restart betting!
        if needs_rebet:
            time.sleep(4.0)  # Give time to read the warning
            for p_num in [1, 2]: self.vision_engine.set_player_overlay(p_num, None)
            self.signals.game_alert.emit("SYSTEM", "Please adjust bets...")
            return  # Restart state_2 without touching any bankrolls

        # --- PASS 2: COMMIT BETS (Safe to deduct) ---
        for i, locked_bet in enumerate([p1_locked_bet, p2_locked_bet]):
            player = self.players[i]
            player_num = i + 1

            # --- 🔥 ALL IN DETECTION (Only if intentional) ---
            is_all_in = (locked_bet > 0 and locked_bet == player["bankroll"])

            if locked_bet == 0:
                self.signals.game_alert.emit(player["name"], "Sitting Out 🪑")
                self.vision_engine.set_player_overlay(player_num, "No Bet Placed\nSitting Out This Round")
                player["current_bet"] = 0
                player["status"] = PlayerStatus.DONE
            else:
                player["current_bet"] = locked_bet
                player["bankroll"] -= locked_bet

                if is_all_in:
                    self.signals.game_alert.emit(player["name"], "ALL IN! 🔥")
                    self.vision_engine.set_player_overlay(player_num, "ALL IN!!!!!!")
                    anyone_all_in = True
                    players_to_clear_video.append(player_num)

            # 5. Update UI
            self.signals.update_bet.emit(player["name"], player["current_bet"])
            self.signals.update_bankroll.emit(player["name"], int(player["bankroll"]))

        # ========================================================
        # POST-LOOP LOGIC
        # ========================================================

        # --- 🔁 REBET LOOPBACK ---
        if needs_rebet:
            time.sleep(4.0)  # Give time to read the warning
            # Clear overlays and return to re-read bets
            for p_num in [1, 2]: self.vision_engine.set_player_overlay(p_num, None)
            self.signals.game_alert.emit("SYSTEM", "Please adjust bets...")
            return  # This restarts the state_2 function from the top

        # --- 🎬 DRAMATIC ALL-IN PAUSE & CLEANUP ---
        if anyone_all_in:
            time.sleep(2.5)

            # BUG FIX: Guaranteed cleanup of All-In overlays before dealing
        for p_num in players_to_clear_video:
            self.vision_engine.set_player_overlay(p_num, None)

        # EMPTY TABLE CHECK
        if p1_locked_bet == 0 and p2_locked_bet == 0:
            time.sleep(3)
            self.current_state = self.STATE_BANKROLL_UPDATE
            return

        # --------------------
        # DEALING PHASE TIMER
        # --------------------
        self.signals.update_phase.emit("INITIAL DEALING")
        self.signals.game_alert.emit("SYSTEM", "Dealer, Deal Initial Cards!")
        self.signals.start_timer_signal.emit(10, "Dealing initial cards...")
        for remaining_seconds in range(10, -1, -1):
            self.signals.update_timer.emit(remaining_seconds)
            if self.current_state != self.STATE_BETTING_DEALING:
                return
            time.sleep(1)

        self.signals.stop_timer_signal.emit()
        self.signals.game_alert.emit("SYSTEM", "Reading Cards...")
        self.camera_thread.clear_buffer()  # Flush the buffer

        # 6. Strict Initial Deal Lock (Wait for exact 2, 2, 1 configuration)
        p1_locked = (self.players[0]["status"] != PlayerStatus.PLAYING)
        p2_locked = (self.players[1]["status"] != PlayerStatus.PLAYING)
        dealer_locked = False

        start_deal_time = time.time()
        timeout_triggered = False

        while not (p1_locked and p2_locked and dealer_locked):
            if getattr(self, 'abort_flag', False):
                return
            if time.time() - start_deal_time > 20:
                self.signals.game_alert.emit("SYSTEM", "Still waiting for missing cards...")
                start_deal_time = time.time()
            current_frames = self.camera_thread.get_recent_frames(num_frames=15)

            # A. Scan all active players
            for i, player in enumerate(self.players):
                if player["status"] == PlayerStatus.PLAYING:
                    is_locked = p1_locked if i == 0 else p2_locked
                    if is_locked:
                        continue
                    roi_key = f"player_{i + 1}_cards"
                    cx1, cy1, cx2, cy2 = self.vision_engine.rois[roi_key]
                    cropped = [f[cy1:cy2, cx1:cx2] for f in current_frames]

                    detected = stabilize_detection(frames_list=cropped,
                                                   detection_func=self.vision_engine.detect_cards)
                    if detected is None:
                        detected = []
                    if len(detected) == 2:
                        for c in detected:
                            player["locked_cards"].append(c)
                            player["hand"].append(c[0])
                            self.game_logic.remove_card_from_inventory(c[0])
                        player["effective_sum"], player["has_ace"] = self.game_logic.calculate_hand_value(
                            player["hand"])
                        if i == 0:
                            p1_locked = True
                        else:
                            p2_locked = True

            # B. Scan Dealer
            if not dealer_locked:
                dx1, dy1, dx2, dy2 = self.vision_engine.rois["dealer_cards"]
                d_cropped = [f[dy1:dy2, dx1:dx2] for f in current_frames]

                d_detected = stabilize_detection(frames_list=d_cropped,
                                                 detection_func=self.vision_engine.detect_cards)
                if d_detected is None:
                    d_detected = []

                if len(d_detected) == 1:
                    for c in d_detected:
                        self.dealer_locked_cards.append(c)
                        self.dealer_hand.append(c[0])
                        self.game_logic.remove_card_from_inventory(c[0])

                    self.dealer_effective_sum, self.dealer_has_ace = self.game_logic.calculate_hand_value(
                        self.dealer_hand)
                    dealer_locked = True

            time.sleep(0.5)

            continue

        d_display_str = self._format_hand_display(self.dealer_effective_sum, self.dealer_has_ace, is_dealer=True)
        self.signals.update_hand.emit("Dealer", d_display_str)

        for player in self.players:
            if player["status"] == PlayerStatus.PLAYING:
                p_display_str = self._format_hand_display(player["effective_sum"], player["has_ace"])
                self.signals.update_hand.emit(player["name"], p_display_str)

        self.signals.game_alert.emit("SYSTEM", "Dealing Complete!")
        time.sleep(1)

        # 9. Initial Blackjack Check (State 2, Steps 66-70)
        for i, player in enumerate(self.players):
            if player["effective_sum"] == 21 and len(player["hand"]) == 2:
                player["status"] = PlayerStatus.BLACKJACK
                self.signals.game_alert.emit(player["name"], "BLACKJACK")
                self.vision_engine.set_player_overlay(i + 1, "BLACKJACK!")

        self.current_state = self.STATE_PLAYER_TURN

    def state_3_player_turn(self) -> None:
        """
        Step 3: Player Loop.
        Iterates through players. Gets strategy, waits for decision (max 25s),
        and deals cards accordingly until the player stands or busts.
        After the turn concludes, calculates final terminal statistics.
        """
        for player_idx, player in enumerate(self.players, start=1):
            if player["current_bet"] == 0:
                continue

            # --- 1. ACTIVE TURN LOOP ---
            while player["status"] == PlayerStatus.PLAYING:
                self.signals.update_phase.emit(f"{player['name']}'s TURN")
                if getattr(self, 'abort_flag', False):
                    return

                if not player["hand"] or player["effective_sum"] == 0:
                    self.signals.update_strategy.emit(player["name"], "WAITING FOR CARDS", None)
                else:
                    # --- UX FIX: Force the UI and Video to refresh before the heavy math freezes the thread ---
                    self.signals.update_strategy.emit(player["name"], "ANALYZING...", None)
                    time.sleep(0.05)  # Releases the GIL so Camera and UI can draw the final clear frame!
                    dealer_upcard = self.dealer_hand[0] if self.dealer_hand else 0
                    strategy_data = self.game_logic.get_strategy_recommendation(
                        player_cards=player["hand"],
                        dealer_upcard_value=dealer_upcard,
                        game_mode=self.game_mode
                    )
                    self.signals.update_strategy.emit(player["name"], strategy_data["action"], strategy_data["stats"])

                self.signals.start_timer_signal.emit(25, "Player decision")
                decision = None

                for remaining_seconds in range(25, -1, -1):
                    if getattr(self, 'abort_flag', False):
                        return
                    self.signals.update_timer.emit(remaining_seconds)
                    time.sleep(1)

                    if remaining_seconds % 2 == 0:
                        recent_frames = self.camera_thread.get_recent_frames(num_frames=15)
                        roi_key = f"player_{player_idx}_decision"
                        decision = stabilize_detection(
                            frames_list=recent_frames,
                            detection_func=self.vision_engine.detect_decision_chip_in_vision,
                            roi_coords=self.vision_engine.rois[roi_key],
                            chosen_player=player_idx
                        )
                        if decision in ["HIT", "STAND", "DOUBLE", "SURRENDER"]:
                            self.signals.stop_timer_signal.emit()
                            break

                if decision not in ["HIT", "STAND", "DOUBLE", "SURRENDER"]:
                    decision = "STAND"
                    self.signals.game_alert.emit(player["name"], "Time's Up! Auto-STAND")
                    self.signals.stop_timer_signal.emit()

                if decision == "HIT":
                    self.signals.game_alert.emit(player["name"], "HIT 🔵")  # UI Emoji
                    success = self.execute_hit(player, player_idx)
                    if not success:
                        self.signals.game_alert.emit(player["name"], "Camera Error. Auto-STAND.")
                        player["status"] = PlayerStatus.DONE

                elif decision == "DOUBLE":
                    can_double = (player["bankroll"] >= player["current_bet"]) and (len(player["hand"]) == 2)
                    if can_double and not player.get("has_doubled", False):
                        self.signals.game_alert.emit(player["name"], "DOUBLE DOWN 🟡")  # UI Emoji

                        player["bankroll"] -= player["current_bet"]
                        player["current_bet"] *= 2
                        player["has_doubled"] = True

                        self.signals.update_bet.emit(player["name"], player["current_bet"])
                        self.signals.update_bankroll.emit(player["name"], int(player["bankroll"]))
                        success = self.execute_hit(player, player_idx)
                        if success and player["status"] == PlayerStatus.PLAYING:
                            self.signals.game_alert.emit(player["name"], "STAND 🔴")  # UI Emoji
                            player["status"] = PlayerStatus.DONE
                    else:
                        reason = "Insufficient Funds" if player["bankroll"] < player[
                            "current_bet"] else "Only allowed on initial hand"
                        self.signals.game_alert.emit("SYSTEM", f"Double rejected: {reason}.\nPlease remove chip!")
                        time.sleep(5)
                        continue

                elif decision == "SURRENDER":
                    if len(player["hand"]) == 2:
                        self.signals.game_alert.emit(player["name"], "🏳️ SURRENDERED")  # UI Emoji
                        player["status"] = PlayerStatus.SURRENDERED
                        self.vision_engine.set_player_overlay(player_idx, "SURRENDERED")  # Clean video text
                    else:
                        self.signals.game_alert.emit("SYSTEM", "Surrender rejected: only allowed on initial hand."
                                                               "\nPlease remove chip!")
                        time.sleep(5)
                        continue

                elif decision == "STAND":
                    player["status"] = PlayerStatus.DONE
                    self.signals.game_alert.emit(player["name"], "🔴 STAND")  # UI Emoji

            # --- 2. END OF TURN LOGIC (Terminal Stats) ---
            terminal_stats = None

            # Evaluate final stats for players who finished their turn (Skip Surrenders)
            if player["status"] != PlayerStatus.SURRENDERED and self.game_mode == "Super Computer":
                dealer_upcard = self.dealer_hand[0] if self.dealer_hand else 0
                terminal_stats = self.game_logic.get_terminal_stats(
                    player_effective_sum=player["effective_sum"],
                    dealer_upcard_value=dealer_upcard,
                    player_status=player["status"].value
                )

            # Always emit the final status!
            # This ensures the UI receives "SURRENDERED" / "BUSTED" / "DONE" and HIDES the strategy widget.
            self.signals.update_strategy.emit(player["name"], player["status"].value, terminal_stats)

            # Show the final hand score
            final_display = self._format_hand_display(player["effective_sum"], player["has_ace"], is_final=True)
            self.signals.update_hand.emit(player["name"], final_display)

        # Once all players are processed
        self.current_state = self.STATE_DEALER_TURN

    def execute_hit(self, player: dict, p_idx: int) -> bool:
        """Helper method to execute a card draw using STRICT Centroid Spatial Tracking (+1)."""
        msg = f"Dealing a card to {player['name']}"
        self.signals.game_alert.emit("SYSTEM", msg)
        self.signals.start_timer_signal.emit(5, msg)

        # PHYSICAL ACTION BUFFER
        # Give dealer time to physically place the single card
        for remaining in range(5, 0, -1):
            self.signals.update_timer.emit(remaining)
            time.sleep(1)
        self.signals.stop_timer_signal.emit()

        # Flush the buffer from moving hands
        self.camera_thread.clear_buffer()

        self.signals.game_alert.emit(player["name"], "Waiting for a new card...")
        new_card_confirmed = False
        start_time = time.time()

        while not new_card_confirmed:
            if getattr(self, 'abort_flag', False):
                return False

            # Timeout protection
            if time.time() - start_time > 20:
                self.signals.game_alert.emit(player["name"], "Still waiting for a card...")
                start_time = time.time()
            time.sleep(0.5)
            card_frames = self.camera_thread.get_recent_frames(num_frames=15)
            roi_key_cards = f"player_{p_idx}_cards"
            hx1, hy1, hx2, hy2 = self.vision_engine.rois[roi_key_cards]
            hit_cropped = [f[hy1:hy2, hx1:hx2] for f in card_frames]
            detected_cards = stabilize_detection(
                frames_list=hit_cropped,
                detection_func=self.vision_engine.detect_cards
            )
            if detected_cards is None:
                detected_cards = []

            # 1. Filter out cards we already know about using Spatial Tracking
            newly_placed_cards = self.get_truly_new_cards_by_distance(
                locked_cards=player["locked_cards"],
                detected_cards=detected_cards,
                distance_threshold=50 * self.vision_engine.scale
            )

            # 2. ENFORCEMENT: Only accept if EXACTLY 1 new physical card is found
            if len(newly_placed_cards) == 1:
                card_data = newly_placed_cards[0]
                player["locked_cards"].append(card_data)

                card_val = card_data[0]
                player["hand"].append(card_val)
                self.game_logic.remove_card_from_inventory(card_val)

                player["effective_sum"], player["has_ace"] = self.game_logic.calculate_hand_value(player["hand"])
                display_str = self._format_hand_display(player["effective_sum"], player["has_ace"])
                self.signals.update_hand.emit(player["name"], display_str)

                self.signals.game_alert.emit("SYSTEM", "")

                # Process bust or auto-stand on 21
                if player["effective_sum"] > 21:
                    player["status"] = PlayerStatus.BUSTED
                    self.signals.game_alert.emit(player["name"], "💥 BUSTED!")  # UI Emoji
                    self.vision_engine.set_player_overlay(p_idx, "BUSTED!")  # Clean video text
                elif player["effective_sum"] == 21:
                    self.signals.game_alert.emit(player["name"], "🎯 21! Auto-STAND")  # UI Emoji
                    player["status"] = PlayerStatus.DONE
                else:
                    self.signals.game_alert.emit(player["name"], "")

                return True
            else:
                # If 0 or 2+ new cards seen, ignore this frame and try again
                time.sleep(1.5)

        return False

    def state_4_dealer_turn(self) -> None:
        """
        Step 4: Dealer Loop.
        The dealer reveals their face-down card and automatically draws
        until reaching an effective sum of 17 or higher.
        """
        # 1. Filter out busted or surrendered players to see who is still 'active'
        active_players = [
            p for p in self.players
            if p["current_bet"] > 0 and p["status"] not in (PlayerStatus.BUSTED, PlayerStatus.SURRENDERED)
        ]

        # If no active players, skip dealer turn entirely
        if not active_players:
            self.signals.game_alert.emit("Dealer", "All players out. Skipping dealer turn.")
            time.sleep(5)
            self.current_state = self.STATE_COMPARISON
            return

        self.signals.update_phase.emit("DEALER's TURN")

        # 2. Check if ALL active players have a Natural Blackjack
        all_active_have_bj = all(p["status"] == PlayerStatus.BLACKJACK for p in active_players)

        if all_active_have_bj:
            dealer_upcard = self.dealer_hand[0] if self.dealer_hand else 0

            # Condition A: Dealer does NOT have 10 or Ace. No chance for Blackjack.
            if dealer_upcard not in [1, 10]:
                self.signals.game_alert.emit("Dealer", "Players have Blackjack. Dealer pays!")
                time.sleep(3)
                self.current_state = self.STATE_COMPARISON
                return

            # Condition B: Dealer HAS 10 or Ace. Must draw exactly ONE card to check for Blackjack.
            else:
                self.signals.game_alert.emit("Dealer", "Checking for Dealer Blackjack...")

        self.signals.game_alert.emit("Dealer", "Dealer's Turn")

        # Loop continues as long as the dealer hasn't reached the stopping condition (17+)
        while self.dealer_effective_sum < 17:

            # Limit dealer to exactly 2 cards if everyone has Blackjack
            if all_active_have_bj and len(self.dealer_hand) >= 2:
                break

            if getattr(self, 'abort_flag', False):
                return

            self.signals.game_alert.emit("Dealer", "Dealing Dealer card...")

            # PHYSICAL ACTION BUFFER
            # Give dealer time to place/flip the card
            for remaining in range(5, 0, -1):
                self.signals.update_timer.emit(remaining)
                time.sleep(1)
            self.signals.stop_timer_signal.emit()

            # Flush the buffer from moving hands
            self.camera_thread.clear_buffer()

            self.signals.game_alert.emit("Dealer", "Waiting for a new card...")
            new_card_confirmed = False
            start_time = time.time()

            while not new_card_confirmed:
                # Timeout protection
                if time.time() - start_time > 20:
                    self.signals.game_alert.emit("Dealer", "Still waiting for a card...")
                    start_time = time.time()

                # 1. Capture frames and detect all currently visible dealer cards
                card_frames = self.camera_thread.get_recent_frames(num_frames=15)
                d4x1, d4y1, d4x2, d4y2 = self.vision_engine.rois["dealer_cards"]
                d4_cropped = [f[d4y1:d4y2, d4x1:d4x2] for f in card_frames]
                detected_cards = stabilize_detection(
                    frames_list=d4_cropped,
                    detection_func=self.vision_engine.detect_cards
                )
                if detected_cards is None:
                    detected_cards = []

                # 2. Filter using Spatial Centroid Tracking
                newly_placed_cards = self.get_truly_new_cards_by_distance(
                    locked_cards=self.dealer_locked_cards,
                    detected_cards=detected_cards,
                    distance_threshold=50 * self.vision_engine.scale
                )

                # 3. ENFORCEMENT: Dealer flips/draws exactly one card at a time
                if len(newly_placed_cards) == 1:
                    card_data = newly_placed_cards[0]
                    self.dealer_locked_cards.append(card_data)

                    card_val = card_data[0]
                    self.dealer_hand.append(card_val)
                    self.game_logic.remove_card_from_inventory(card_val)

                    # 4. Recalculate sum and update UI
                    self.dealer_effective_sum, self.dealer_has_ace = self.game_logic.calculate_hand_value(
                        self.dealer_hand)
                    display_str = self._format_hand_display(self.dealer_effective_sum, self.dealer_has_ace,
                                                            is_final=True, is_dealer=True)
                    self.signals.update_hand.emit("Dealer", display_str)

                    if self.game_mode == "Super Computer":
                        if self.dealer_effective_sum < 17:
                            current_probs = self.game_logic.get_dealer_forecast(self.dealer_hand)
                            # Send signal specifically for the dealer forecast
                            self.signals.update_strategy.emit("Dealer", "UPDATE", {"dealer_probs": current_probs})

                    new_card_confirmed = True
                else:
                    # If 0 or 2+ new cards seen, wait and try again
                    time.sleep(1.5)

        # 5. Check final Dealer status after exiting the loop
        dealer_has_bj = (len(self.dealer_hand) == 2 and self.dealer_effective_sum == 21)

        if self.dealer_effective_sum > 21:
            self.signals.game_alert.emit("Dealer", "BUSTED")
        elif all_active_have_bj and not dealer_has_bj:
            self.signals.game_alert.emit("Dealer", "No Blackjack! Players Win 💸")
        else:
            self.signals.game_alert.emit("Dealer", f"Stands on {self.dealer_effective_sum}")
        time.sleep(2)

        # 6. Transition to determine winners
        self.current_state = self.STATE_COMPARISON

    def state_5_comparison(self) -> None:
        """
        Step 5: Determine Winners.
        Compares final player hands against dealer hand.
        """
        for player in self.players:
            if player["current_bet"] == 0:
                continue

            dealer_has_bj = (len(self.dealer_hand) == 2 and self.dealer_effective_sum == 21)
            p_idx = self.players.index(player) + 1

            # 1. CASE: Player already Busted in State 3
            if player["status"] == PlayerStatus.BUSTED:
                player["status"] = PlayerStatus.LOSE
                self.signals.game_alert.emit(player["name"], "💥 Bust - Lose")
                time.sleep(2)
                continue

            # 2. CASE: Player has Natural Blackjack
            if player["status"] == PlayerStatus.BLACKJACK:
                if dealer_has_bj:
                    player["status"] = PlayerStatus.PUSH
                    self.signals.game_alert.emit(player["name"], "🤝 Blackjack - Push (Tie)")
                    self.vision_engine.set_player_overlay(p_idx, "PUSH")
                else:
                    self.signals.game_alert.emit(player["name"], "👑 Blackjack!")
                    self.vision_engine.set_player_overlay(p_idx, "WIN!")
                time.sleep(2)
                continue

            # 3. CASE: Dealer has Natural Blackjack (Player has regular hand)
            if dealer_has_bj:
                player["status"] = PlayerStatus.LOSE
                self.signals.game_alert.emit(player["name"], "💀 Dealer Blackjack - Lose")
                # FIX: Set the video overlay before the continue shortcut
                self.vision_engine.set_player_overlay(p_idx, "LOSE")
                time.sleep(2.5)
                continue

            # 4. CASE: Player Surrendered
            if player["status"] == PlayerStatus.SURRENDERED:
                self.signals.game_alert.emit(player["name"], "🏳️ Surrendered")
                # Note: "SURRENDERED" overlay was already set during decision detection
                time.sleep(2.5)
                continue

            # 5. CASE: Normal Hand Comparison
            if self.dealer_effective_sum > 21:
                player["status"] = PlayerStatus.WIN
            elif player["effective_sum"] > self.dealer_effective_sum:
                player["status"] = PlayerStatus.WIN
            elif player["effective_sum"] < self.dealer_effective_sum:
                player["status"] = PlayerStatus.LOSE
            else:
                player["status"] = PlayerStatus.PUSH

            # Emit UI Emoji Label
            label_ui = {"WIN": "Win! 🏆", "LOSE": "Lose 💀", "PUSH": "Push - Tie 🤝"}.get(
                player["status"].value, player["status"].value
            )
            self.signals.game_alert.emit(player["name"], label_ui)

            # Emit Clean Video Overlay Label (OpenCV)
            label_video = {"WIN": "WIN!", "LOSE": "LOSE", "PUSH": "PUSH"}.get(
                player["status"].value, player["status"].value
            )
            self.vision_engine.set_player_overlay(p_idx, label_video)

            time.sleep(3.5)  # Let Player read the result before moving to next player

        self.current_state = self.STATE_BANKROLL_UPDATE

    def state_6_bankroll_update(self) -> None:
        """
        Step 6: Update Bankroll & Trigger End of Round UI.
        Calculates Gross Payout (Physical chips returned to player).
        """
        for player in self.players:
            status = player["status"].value
            bet = player["current_bet"]
            payout = 0

            # --- Calculate Gross Payout (What the dealer pushes back) ---
            if status == "WIN":
                payout = bet * 2
            elif status == "BLACKJACK":
                payout = int(bet * 2.5)
            elif status == "PUSH":
                payout = bet
            elif status == "SURRENDERED":
                payout = bet // 2
            elif status in ["LOSE", "BUSTED"]:
                payout = 0

            # Emit the physical payout and the original bet for color logic
            if bet > 0:
                self.signals.financial_result.emit(player["name"], payout, bet)

            # Mathematical update (Bankroll already deducted the bet in State 2)
            player["bankroll"] = self.game_logic.bank_roll_update(
                current_bankroll=player["bankroll"],
                current_bet=player["current_bet"],
                match_result=status
            )
            self.signals.update_bankroll.emit(player["name"], int(player["bankroll"]))

        self.signals.update_phase.emit("ROUND OVER")
        self.signals.game_alert.emit("SYSTEM", "Please clear cards")
        self.signals.round_over_signal.emit()
        self.current_state = self.STATE_WAIT_FOR_NEXT_ROUND

    def state_7_wait_for_next_round(self) -> None:
        """
        Step 7: Idle State.
        The game loop simply idles here while waiting for the UI buttons to be clicked.
        The UI buttons will trigger callbacks that change self.current_state.
        """
        pass  # Do nothing. Allow UI to be responsive.

    # --- Callbacks for the UI Buttons (Add these to GameManager) ---
    def on_game_start(self, data: dict) -> None:
        """
        Triggered by the UI game_start_request signal after calibration.
        Populates players, sets deck config, and starts the actual game!
        """
        if self.camera_thread:
            recent_frames = self.camera_thread.get_recent_frames(num_frames=1)
            if recent_frames:
                self.vision_engine.calibrate_lighting(recent_frames[0])

        # 1. Update the number of decks in Logic before starting the game
        num_decks = data.get("num_decks", 2)
        self.game_logic.set_game_config(num_decks)

        if self.ui_manager and hasattr(self.ui_manager, 'house_rules_widget'):
            self.ui_manager.house_rules_widget.update_rules(num_decks, self.game_logic.shuffle_threshold)

        # 2. Initialize players
        self.players = [
            {"name": data["p1_name"], "bankroll": data["p1_bankroll"], "current_bet": 0, "hand": [],
             "locked_cards": [], "has_ace": False, "status": PlayerStatus.PLAYING, "has_doubled": False},
            {"name": data["p2_name"], "bankroll": data["p2_bankroll"], "current_bet": 0, "hand": [],
             "locked_cards": [], "has_ace": False, "status": PlayerStatus.PLAYING, "has_doubled": False}
        ]

        self.dealer_locked_cards = []
        self.game_mode = data["mode"]
        self.current_state = self.STATE_BETTING_DEALING
        print(f"[GAME MANAGER] Started with {num_decks} decks. Mode: {self.game_mode}")

    def on_new_round_clicked(self) -> None:
        """ Triggered by the UI 'New Round' button """
        self.vision_engine.clear_player_overlays()
        self.abort_flag = True
        self.current_state = self.STATE_BETTING_DEALING

    def on_new_game_clicked(self) -> None:
        """
        Triggered by the UI 'New Game' button.
        Places the game manager in the INIT state, waiting for new config from UI.
        """
        self.vision_engine.clear_player_overlays()
        self.abort_flag = True
        self.current_state = self.STATE_INIT_GAME
        self.signals.game_alert.emit("SYSTEM", "Setup New Game...")

    def on_shuffle_complete_clicked(self) -> None:
        """
        Triggered by the UI when the dealer physically finishes shuffling.
        Mathematically resets the logic inventory and resumes the game.
        """
        # Automatically updates exactly to the configured num_decks!
        self.game_logic.reshuffle_decks()

        if self.ui_manager and hasattr(self.ui_manager, 'house_rules_widget'):
            self.ui_manager.house_rules_widget.update_rules(self.game_logic.num_decks,
                                                            self.game_logic.shuffle_threshold)

        # Clear the massive video overlay
        self.vision_engine.set_system_overlay(None)

        # Hide the button and resume State 2 (Betting and Dealing)
        self.signals.show_shuffle_button.emit(False)
        self.signals.game_alert.emit("SYSTEM", "Cards Ready! Resuming...")
        self.current_state = self.STATE_BETTING_DEALING

    def get_truly_new_cards_by_distance(self, locked_cards: List[List[int]], detected_cards: List[List[int]],
                                        distance_threshold: int = 40) -> List[List[int]]:
        """
        Centroid Tracking Filter: Compares newly detected cards against already 'locked' cards.
        If a detected card is physically too close to an existing one (overlap), it is ignored.

        Args:
            locked_cards: Cards already in hand [[val, cx, cy], ...]
            detected_cards: Newly processed frame cards [[val, cx, cy], ...]

        Returns:
            List[List[int]]: Only the truly new cards that were placed on the table.
        """
        new_valid_cards = []

        for det_card in detected_cards:
            # Ensure proper format from vision engine [val, cx, cy]
            if not det_card or len(det_card) != 3:
                continue

            det_val, det_cx, det_cy = det_card
            is_overlap = False

            for locked_card in locked_cards:
                lock_val, lock_cx, lock_cy = locked_card

                # Calculate Euclidean distance between centers
                distance = math.hypot(det_cx - lock_cx, det_cy - lock_cy)

                if distance < distance_threshold:
                    is_overlap = True
                    break

            if not is_overlap:
                new_valid_cards.append(det_card)

        return new_valid_cards

    def _format_hand_display(self, effective_sum: int, is_soft: bool, is_final: bool = False,
                             is_dealer: bool = False) -> str:
        """
        Formats the hand value for UI display.
        If it's a soft hand AND the player is still deciding, shows 'Low/High' (e.g., '5/15').
        If it's the dealer or a player who finished their turn (is_final=True), shows strict integer.
        """
        if effective_sum == 0:
            return "0"

        # Check for Natural Blackjack (Dealer specific request)
        if is_dealer and effective_sum == 21 and len(self.dealer_hand) == 2:
            return "BLACKJACK"

        if is_soft and effective_sum < 21 and not is_final:
            low_val = effective_sum - 10
            return f"{low_val}/{effective_sum}"

        return str(effective_sum)
