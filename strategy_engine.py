### The final version for Github ###

from typing import Dict, Tuple, Any, List


def get_effective_hand(player_hard_sum: int, has_ace: bool) -> Tuple[int, bool]:
    """
    Calculates the maximum valid hand value given the hard sum and number of aces.

    Returns:
        Tuple[int, bool]: (effective_sum, is_soft_hand)

    True = soft_hand , False = hard_hand
    """
    if has_ace and (player_hard_sum + 10) <= 21:
        return player_hard_sum + 10, True
    return player_hard_sum, False


def hashable_deck(deck_state: Dict[int, int]) -> Tuple:
    """
    Converts the deck state dictionary (Mutable) into a hashable tuple (Immutable).
    Required for using the deck state as a dictionary key in the cache.
    """
    return tuple(sorted(deck_state.items()))


# ==============================================
# 1. The Perfect Gambler - Static Table Strategy
# ==============================================
def best_gambler_strategy_recommendation(player_cards: List[int], dealer_upcard: int, num_decks: int = 2) -> str:
    """
    Lookup function for Blackjack Basic Strategy.
    Acts as a static deterministic table and ignores deck composition.
    Includes rules for Stand, Hit, Double Down, and Surrender.

    Arguments:
        player_cards (List[int]): A list containing the raw numeric values of the player's current cards.
        dealer_upcard (int): The numeric value of the dealer's visible card (1-10).

    Returns:
        str: The final recommended action ('Hit', 'Stand', 'Double', 'Surrender', 'Error: Invalid Dealer Card',
        'Error: Empty Player Hand', 'Error: Player Already Busted').
    """

    # 1. Parse hand state
    is_initial_hand = len(player_cards) == 2
    hard_sum = sum(player_cards)
    has_ace = 1 in player_cards

    # 2. Edge cases validation
    if dealer_upcard < 1 or dealer_upcard > 10:
        return 'Error: Invalid Dealer Card'

    if hard_sum < 1:
        return 'Error: Empty Player Hand'

    if hard_sum > 21:
        return 'Error: Player Already Busted'

    # 3. Calculate effective sum and hand type
    eff_sum, is_soft = get_effective_hand(hard_sum, has_ace)

    # 4. Terminal state: Automatic Stand on 21
    if eff_sum == 21:
        return 'Stand'

    # 5. SURRENDER LOGIC (Only valid on initial 2-card hand)
    if is_initial_hand:
        if not is_soft:
            # Hard 16 surrenders against 9, 10, or Ace (1)
            if eff_sum == 16 and dealer_upcard in [9, 10, 1]:
                return 'Surrender'
            # Hard 15 surrenders against a 10
            if eff_sum == 15 and dealer_upcard == 10:
                return 'Surrender'

    action = 'Stand'

    # 6. HIT, STAND, AND DOUBLE DOWN LOGIC
    if is_soft:
        # SOFT HANDS LOGIC
        if eff_sum in [13, 14]:
            if is_initial_hand and dealer_upcard in [5, 6]:
                action = 'Double'
            else:
                action = 'Hit'
        elif eff_sum in [15, 16]:
            if is_initial_hand and dealer_upcard in [4, 5, 6]:
                action = 'Double'
            else:
                action = 'Hit'
        elif eff_sum == 17:
            if is_initial_hand and dealer_upcard in [3, 4, 5, 6]:
                action = 'Double'
            else:
                action = 'Hit'
        elif eff_sum == 18:
            if is_initial_hand and dealer_upcard in [3, 4, 5, 6]:
                action = 'Double'
            elif dealer_upcard in [2, 7, 8]:
                action = 'Stand'
            else:
                action = 'Hit'
        elif eff_sum >= 19:
            action = 'Stand'

    else:
        # HARD HANDS LOGIC
        if eff_sum <= 8:
            action = 'Hit'
        elif eff_sum == 9:
            if is_initial_hand and dealer_upcard in [3, 4, 5, 6]:
                action = 'Double'
            else:
                action = 'Hit'
        elif eff_sum == 10:
            if is_initial_hand and dealer_upcard in [2, 3, 4, 5, 6, 7, 8, 9]:
                action = 'Double'
            else:
                action = 'Hit'
        elif eff_sum == 11:
            if is_initial_hand:
                if num_decks == 1:
                    action = 'Double'
                else:
                    # Double against anything except an Ace (1)
                    if dealer_upcard in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        action = 'Double'
                    else:
                        action = 'Hit'
            else:
                action = 'Hit'
        elif eff_sum == 12:
            if dealer_upcard in [4, 5, 6]:
                action = 'Stand'
            else:
                action = 'Hit'
        elif 13 <= eff_sum <= 16:
            if dealer_upcard in [2, 3, 4, 5, 6]:
                action = 'Stand'
            else:
                action = 'Hit'
        elif eff_sum >= 17:
            action = 'Stand'

    return action


# =============================================================
# 2. Super Computer - Dynamic EV & Card Counting Strategy (S17)
# =============================================================

# Memoization caches
dealer_memo: Dict[Tuple, Dict[Any, float]] = {}
player_memo: Dict[Tuple, Dict[str, Any]] = {}


def get_dealer_probs(hard_sum: int, has_ace: bool, deck_state: Dict[int, int]) -> Dict[Any, float]:
    """
    Recursively calculates the final probability distribution of the dealer's hand
    following the S17 (Stands on Soft 17) rule.
    """
    state_key = (hard_sum, has_ace, hashable_deck(deck_state))

    # Checking if the state is in the dealer's cache memory
    if state_key in dealer_memo:
        return dealer_memo[state_key]

    total_cards = sum(deck_state.values())

    # Edge case: Deck is empty (return final outcome to keep probabilities valid and prevents division by zero)
    if total_cards == 0:
        eff_sum, _ = get_effective_hand(hard_sum, has_ace)
        res = {'Bust': 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0}
        if eff_sum > 21:
            res['Bust'] = 1.0
        else:
            res[eff_sum] = 1.0
        return res

    eff_sum, _ = get_effective_hand(hard_sum, has_ace)

    # Base case: Dealer reaches 17 or higher (S17 rule)
    if eff_sum >= 17:
        if eff_sum > 21:
            return {'Bust': 1.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0}
        else:
            res = {'Bust': 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0}
            res[eff_sum] = 1.0
            return res

    # Recursive case: Dealer must hit
    probs = {'Bust': 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0}
    for card_val, count in deck_state.items():
        if count > 0:  # Iterate over the 10 card types, only those currently in stock
            prob = count / total_cards

            # Create the alternative reality where this card is drawn
            deck_state[card_val] -= 1
            new_hard = hard_sum + card_val
            new_has_ace = has_ace or (card_val == 1)

            # Traverse the branch and accumulate probabilities
            branch_probs = get_dealer_probs(new_hard, new_has_ace, deck_state)  # The recursive call

            # BACKTRACKING: Revert state
            deck_state[card_val] += 1

            for k in probs:
                # Law of Total Probability: multiply the card's prob by the branch results and add to the total sum
                probs[k] += prob * branch_probs[k]

    # Store the dealer's final outcome distribution (Bust, 17-21) for the current state
    dealer_memo[state_key] = probs
    return probs


def get_stand_probs(player_eff_sum: int, dealer_upcard: int, deck_state: Dict[int, int]) -> Dict[str, float]:
    """
    Calculates the player's win, loss and tie probabilities if the action is 'Stand'.
    Compares the player's final effective sum against the dealer's simulated outcomes.

    Arguments:
        player_eff_sum (int): The player's highest valid hand sum (up to 21).
        dealer_upcard (int): The numeric value of the dealer's visible card (1-10).
        deck_state (Dict[int, int]): A dictionary representing the current state of the deck.

    Returns:
        Dict[str, float]: A dictionary with probabilities for 'win', 'loss', and 'tie'.
    """
    dealer_has_aces = (dealer_upcard == 1)

    # Standard execution without the peek logic:
    dealer_probs = get_dealer_probs(dealer_upcard, dealer_has_aces, deck_state)

    results = {'win': 0.0, 'loss': 0.0, 'tie': 0.0}

    for outcome, prob in dealer_probs.items():
        if prob == 0:
            continue

        # Compare dealer's final outcome against player's hand
        if outcome == 'Bust' or player_eff_sum > outcome:
            results['win'] += prob
        elif player_eff_sum < outcome:
            results['loss'] += prob
        else:
            results['tie'] += prob

    return results


def super_computer_strategy_recommendation(player_cards: List[int], dealer_upcard: int, deck_state: Dict[int, int]) -> \
Dict[str, Any]:
    """
    Dynamic Blackjack decision engine based on Expected Value (EV).
    Calculates the mathematically optimal action by recursively simulating all future
    dealer and player outcomes using the exact remaining deck composition.

    Arguments:
        player_cards (List[int]): A list of numeric values representing the player's current hand
                                 (e.g., [1, 7] for a Soft 18).
        dealer_upcard (int): The numeric value of the dealer's visible card (1-10).
        deck_state (Dict[int, int]): A dictionary representing the current state of the shoe
                                     (Key: card value 1-10, Value: remaining count).

    Returns:
        Dict[str, Any]: A dictionary containing the decision and detailed statistical breakdown:
            - 'action' (str): The recommended move or current state. Possible values:
                * 'Hit', 'Stand', 'Double', 'Surrender': Standard mathematically optimal plays.
                * 'Bust': Returned if the calculated effective sum exceeds 21.
                * 'Error: Invalid Dealer Card': If dealer_upcard is not between 1 and 10.
                * 'Error: Empty Player Hand': If the player_cards list is empty.
                * 'Error: Player Already Busted': If the input hand is already over 21.
                * 'Error: Negative Card Count in Deck': If deck_state contains counts below 0.
            - 'ev' (float): The Expected Value (profit/loss) for the recommended action.
            - 'hit_stats' (Dict[str, float]): Win/Loss/Tie probabilities if 'Hit' is selected.
            - 'stand_stats' (Dict[str, float]): Win/Loss/Tie probabilities if 'Stand' is selected.
            - 'double_stats' (Dict[str, float] | None): Probabilities for 'Double', or None if the
                                                        hand has more than 2 cards.
    """

    # 1. Extract hand properties dynamically from the list
    is_initial_hand = len(player_cards) == 2
    hard_sum = sum(player_cards)
    has_ace = 1 in player_cards

    # 2. Edge case: Invalid state with specific error tracking
    if any(count < 0 for count in deck_state.values()):
        error_msg = 'Error: Negative Card Count in Deck'
    elif dealer_upcard < 1 or dealer_upcard > 10:
        error_msg = 'Error: Invalid Dealer Card'
    elif hard_sum < 1:
        error_msg = 'Error: Empty Player Hand'
    else:
        error_msg = None

    # If an error was found, return a structured error dictionary
    if error_msg:
        bad_stats = {'win': 0.0, 'loss': 0.0, 'tie': 0.0}
        return {
            'action': error_msg,
            'all_actions': {
                'HIT': {'ev': 0.0, 'stats': bad_stats},
                'STAND': {'ev': 0.0, 'stats': bad_stats}
            },
            'ev': 0.0,
            'dealer_probs': None
        }

    dealer_has_ace = (dealer_upcard == 1)
    current_dealer_probs = get_dealer_probs(dealer_upcard, dealer_has_ace, deck_state)

    # 3. Cache memory handling (Order of cards doesn't matter)
    state_key = (tuple(sorted(player_cards)), dealer_upcard, hashable_deck(deck_state))

    if state_key in player_memo:
        return player_memo[state_key]

    eff_sum, is_soft = get_effective_hand(hard_sum, has_ace)

    # 4. Edge case: Player Busts
    if eff_sum > 21:
        bad_stats = {'win': 0.0, 'loss': 1.0, 'tie': 0.0}
        return {
            'action': 'Bust',
            'all_actions': {
                'HIT': {'ev': -1.0, 'stats': bad_stats},
                'STAND': {'ev': -1.0, 'stats': bad_stats}
            },
            'ev': -1.0,
            'dealer_probs': current_dealer_probs
        }

    total_cards = sum(deck_state.values())

    # 5. Calculate STAND EV (Terminal Node)
    stand_probs = get_stand_probs(eff_sum, dealer_upcard, deck_state)
    stand_ev = stand_probs['win'] - stand_probs['loss']

    if eff_sum == 21:
        bad_stats = {'win': 0.0, 'loss': 1.0, 'tie': 0.0}
        res = {
            'action': 'Stand',
            'all_actions': {
                'STAND': {'ev': stand_ev, 'stats': stand_probs},
                'HIT': {'ev': -1.0, 'stats': bad_stats}
            },
            'ev': stand_ev,
            'dealer_probs': current_dealer_probs
        }
        player_memo[state_key] = res
        return res

    # 6. Calculate HIT EV (Recursive Node)
    hit_probs = {'win': 0.0, 'loss': 0.0, 'tie': 0.0}

    if total_cards > 0:
        for card_val, count in deck_state.items():
            if count > 0:
                prob = count / total_cards

                # Simulate card draw
                deck_state[card_val] -= 1

                new_cards = player_cards + [card_val]
                new_hard = sum(new_cards)
                new_has_ace = 1 in new_cards
                new_eff_sum, _ = get_effective_hand(new_hard, new_has_ace)

                if new_eff_sum > 21:
                    # Player busts
                    current_card_stats = {'win': 0.0, 'loss': 1.0, 'tie': 0.0}
                else:
                    # Compare immediate Stand EV vs. recursive Hit EV
                    res_if_stand = get_stand_probs(new_eff_sum, dealer_upcard, deck_state)

                    future_res = super_computer_strategy_recommendation(new_cards, dealer_upcard, deck_state)
                    res_if_hit_again = future_res['all_actions']['HIT']['stats']

                    ev_stand = res_if_stand['win'] - res_if_stand['loss']
                    ev_hit_again = res_if_hit_again['win'] - res_if_hit_again['loss']

                    # Select the path that yields the highest Expected Value
                    if ev_stand >= ev_hit_again:
                        current_card_stats = res_if_stand
                    else:
                        current_card_stats = res_if_hit_again

                # Backtracking
                deck_state[card_val] += 1

                hit_probs['win'] += prob * current_card_stats['win']
                hit_probs['loss'] += prob * current_card_stats['loss']
                hit_probs['tie'] += prob * current_card_stats['tie']

    hit_ev = hit_probs['win'] - hit_probs['loss']

    # 7. Calculate DOUBLE DOWN EV (Terminal Node)
    double_probs = {'win': 0.0, 'loss': 0.0, 'tie': 0.0}
    double_ev = -float('inf')

    if is_initial_hand and total_cards > 0:
        for card_val, count in deck_state.items():
            if count > 0:
                prob = count / total_cards

                # Temporarily remove the drawn card from the deck state for the simulation
                deck_state[card_val] -= 1

                final_cards = player_cards + [card_val]
                final_hard = sum(final_cards)
                final_has_ace = 1 in final_cards
                final_eff_sum, _ = get_effective_hand(final_hard, final_has_ace)

                # A Double Down results in exactly one additional card.
                # If the player busts, the wager is lost immediately.
                if final_eff_sum > 21:
                    double_probs['loss'] += prob
                else:
                    # If the player survives the draw, the dealer plays out their hand.
                    final_stand_probs = get_stand_probs(final_eff_sum, dealer_upcard, deck_state)
                    double_probs['win'] += prob * final_stand_probs['win']
                    double_probs['loss'] += prob * final_stand_probs['loss']
                    double_probs['tie'] += prob * final_stand_probs['tie']

                # Restore the deck state for the next iteration to conserve memory
                deck_state[card_val] += 1

        # The Expected Value for a Double Down is multiplied by 2
        # to account for the doubled initial wager.
        double_ev = (double_probs['win'] - double_probs['loss']) * 2.0

    # 8. Calculate SURRENDER EV
    # Surrendering means giving up half the bet. Expected Value is exactly -0.5.
    surrender_ev = -0.5 if is_initial_hand else -float('inf')

    # 9. Final Decision Logic: Find the action with the highest Expected Value
    best_action = 'Stand'
    max_ev = stand_ev

    if hit_ev > max_ev:
        best_action = 'Hit'
        max_ev = hit_ev

    if is_initial_hand and double_ev > max_ev:
        best_action = 'Double'
        max_ev = double_ev

    if is_initial_hand and surrender_ev > max_ev:
        best_action = 'Surrender'
        max_ev = surrender_ev

    # Build the base actions dictionary with mathematically valid moves
    all_acts = {
        'HIT': {'ev': hit_ev, 'stats': hit_probs},
        'STAND': {'ev': stand_ev, 'stats': stand_probs}
    }

    # Only inject DOUBLE and SURRENDER if the rules allow it (Initial 2-card Hand)
    if is_initial_hand:
        all_acts['DOUBLE'] = {'ev': double_ev, 'stats': double_probs}
        all_acts['SURRENDER'] = {'ev': surrender_ev, 'stats': {'win': 0.0, 'loss': 1.0, 'tie': 0.0}}

    res = {
        'action': best_action.upper(),
        'ev': max_ev,
        'all_actions': all_acts,
        'dealer_probs': current_dealer_probs
    }

    # Store the final result in cache
    player_memo[state_key] = res
    return res


# =============================================================
# 3. Evaluate Pre Round Statistics and Bet Recommendation
# =============================================================

# 1-Deck EOR (Effect of Removal) values.
# Represents the percentage shift in EV when ONE card is removed from a SINGLE deck.
EOR_1_DECK = {
    1: -0.60,
    2: 0.38,
    3: 0.45,
    4: 0.55,
    5: 0.68,
    6: 0.46,
    7: 0.28,
    8: 0.00,
    9: -0.16,
    10: -0.51
}

# Base EV (House Edge) changes depending on the total number of decks used (S17 rules)
BASE_EV_BY_DECK = {
    1: 0.02,  # Almost even
    2: -0.32,
    4: -0.48,
    6: -0.54,
    8: -0.57
}


def evaluate_pre_round_stats(deck_state: Dict[int, int], num_decks: int = 2) -> Dict[str, Any]:
    """
    Evaluates the deck composition before any cards are dealt in the new round.
    Dynamically adjusts the mathematical edge based on the total number of decks
    using the Effect of Removal (EOR) principle.

    Arguments:
        deck_state (Dict[int, int]): A dictionary representing the current state of the shoe
                                     (Key: card value 1-10, Value: remaining count).
        num_decks (int): Total number of decks originally used in the shoe (default is 2).

    Returns:
        Dict[str, Any]: A structured dictionary containing the pre-round analysis:
            - 'ev' (float): The player's Expected Value (advantage/disadvantage margin) in percentages.
            - 'bet_recommendation' (str): Recommended betting action based on Kelly Criterion logic.
                                          Values: 'High Bet', 'Increase Bet', 'Minimum Bet', or 'Wait for shuffle'.
            - 'player_stats' (Dict[str, float]): Estimated win/loss/tie percentages for the player.
                * Keys: 'win_percent', 'loss_percent', 'tie_percent'.
            - 'dealer_stats' (Dict[Any, float]): Exact probabilities for the dealer's final outcome
                                                 calculated across all possible upcards.
                * Keys: 'Bust' (str), 17, 18, 19, 20, 21 (int).
    """
    total_cards = sum(deck_state.values())

    # Edge case: Empty or almost empty shoe (Wait for the dealer to shuffle)
    if total_cards < 10:
        return {
            'ev': 0.0,
            'bet_recommendation': 'Not enough cards for evaluation',
            'player_stats': {'win_percent': 0.0, 'loss_percent': 0.0, 'tie_percent': 0.0},
            'dealer_stats': {'Bust': 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0}
        }

    # =========================================================================
    # 1. Calculate Expected Value (EV) dynamically based on num_decks
    # =========================================================================
    # Fallback to 2-deck baseline if an irregular number of decks is provided
    base_ev = BASE_EV_BY_DECK.get(num_decks, -0.32)
    base_win = 42.22
    base_loss = 49.10
    base_tie = 8.68

    total_ev_shift_1_deck = 0.0

    # Calculate how missing cards affect the math
    for card_val in range(1, 11):
        cards_per_deck = 16 if card_val == 10 else 4
        expected_total_count = num_decks * cards_per_deck
        current_count = deck_state.get(card_val, 0)

        missing_count = expected_total_count - current_count

        # Accumulate the raw 1-deck shifts
        total_ev_shift_1_deck += missing_count * EOR_1_DECK[card_val]

    # Dilute the effect based on the number of decks in the shoe
    actual_ev_shift = total_ev_shift_1_deck / num_decks
    final_ev = base_ev + actual_ev_shift

    # Estimate player win/loss based on the EV shift (1% EV shift = ~0.5% win shift)
    shift_ratio = actual_ev_shift / 2.0
    est_win = base_win + shift_ratio
    est_loss = base_loss - shift_ratio

    # =========================================================================
    # 2. Bet Recommendation Logic - Kelly Criterion
    # =========================================================================
    if final_ev > 1.5:
        bet_rec = '🔥 MAX BET 🔥'
    elif final_ev > 0.0:
        bet_rec = '📈 RAISE BET 📈'
    else:
        bet_rec = '🛡️ MINIMUM BET 🛡️'

    # =========================================================================
    # 3. Calculate Exact Dealer Probabilities for the upcoming round
    # =========================================================================
    dealer_overall_stats = {'Bust': 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0}

    for upcard, count in deck_state.items():
        if count > 0:
            prob_of_upcard = count / total_cards

            # Simulate the dealer pulling this specific upcard
            deck_state[upcard] -= 1
            has_ace = (upcard == 1)

            # Fetch exact future probabilities using the S17 recursive function
            upcard_stats = get_dealer_probs(upcard, has_ace, deck_state)

            # Revert
            deck_state[upcard] += 1

            # Weight the outcome by the probability of the upcard appearing
            for outcome in dealer_overall_stats:
                dealer_overall_stats[outcome] += prob_of_upcard * upcard_stats.get(outcome, 0.0)

    # Format output for the UI
    return {
        'ev': round(final_ev, 2),
        'bet_recommendation': bet_rec,
        'player_stats': {
            'win': round(est_win / 100.0, 4),
            'loss': round(est_loss / 100.0, 4),
            'tie': round(base_tie / 100.0, 4)
        },
        'dealer_probs': {k: round(v * 100, 2) for k, v in dealer_overall_stats.items()}
    }


def clear_strategy_cache() -> None:
    """Clears the global memoization dictionaries to prevent memory leaks."""
    global dealer_memo, player_memo
    dealer_memo.clear()
    player_memo.clear()
    print("[STRATEGY ENGINE] Cache Cleared to prevent memory leak.")


if __name__ == "__main__":

    # =========================================================================
    # 1. TEST RUNNER: STATIC TABLE
    # =========================================================================
    def test_static_table_decision(player_cards: List[int], dealer_upcard: int) -> str:
        # Calculate effective sum for clean display
        hard_sum = sum(player_cards)
        has_ace = 1 in player_cards
        eff_sum, is_soft = get_effective_hand(hard_sum, has_ace)
        hand_type = "Soft" if is_soft else "Hard"

        # Get recommendation from the updated static table
        action = best_gambler_strategy_recommendation(player_cards, dealer_upcard)

        print(f"[STATIC TABLE] Cards: {player_cards} -> {eff_sum} ({hand_type}) | Dealer: {dealer_upcard}")
        print(f"RECOMMENDATION: 🔥 {action.upper()} 🔥\n")

        return action


    print("\n" + "=" * 40)
    print("=== TESTING STATIC TABLE ===")
    print("=" * 40)

    # Test 1: Hard 16 vs 10 (Expected: Surrender)
    test_static_table_decision(player_cards=[10, 6], dealer_upcard=10)

    # Test 2: Hard 11 vs 6 (Expected: Double)
    test_static_table_decision(player_cards=[8, 3], dealer_upcard=6)

    # Test 3: Soft 18 vs 9 (Expected: Hit)
    test_static_table_decision(player_cards=[1, 7], dealer_upcard=9)

    # Test 4: Player exceeds 21 (Expected: Error: Player Already Busted)
    test_static_table_decision(player_cards=[10, 8, 5], dealer_upcard=5)

    # =========================================================================
    # 2. TEST RUNNER: DYNAMIC SUPER COMPUTER
    # =========================================================================
    def get_decision_from_state(deck_state: Dict[int, int], player_cards: List[int], dealer_upcard: int) -> \
            Dict[str, Any]:
        result = super_computer_strategy_recommendation(player_cards, dealer_upcard, deck_state)

        print(f"--- INPUT ---")
        print(f"Cards: {player_cards} | Dealer Upcard: {dealer_upcard}")

        # Print formatted output
        action = result['action']
        ev = result['ev']

        # Pull stats from the new 'all_actions' dictionary
        hit = result['all_actions']['HIT']['stats']
        stand = result['all_actions']['STAND']['stats']

        double_data = result['all_actions'].get('DOUBLE')
        double = double_data['stats'] if double_data else None

        print(f"--- FORMATTED OUTPUT ---")
        print(f"Recommendation: {action.upper()} (EV: {ev:+.3f})")
        print(
            f"Hit      -> Win: {hit['win'] * 100:.1f}% | Tie: {hit['tie'] * 100:.1f}% | Loss: {hit['loss'] * 100:.1f}%")
        print(
            f"Stand    -> Win: {stand['win'] * 100:.1f}% | Tie: {stand['tie'] * 100:.1f}% | Loss: {stand['loss'] * 100:.1f}%")

        if double:
            print(
                f"Double   -> Win: {double['win'] * 100:.1f}% | Tie: {double['tie'] * 100:.1f}% | Loss: {double['loss'] * 100:.1f}%")
        else:
            print("Double   -> Not Available")

        print("-" * 25 + "\n")
        return result


    def get_test_deck() -> Dict[int, int]:
        return {1: 8, 2: 6, 3: 6, 4: 7, 5: 7, 6: 8, 7: 8, 8: 8, 9: 8, 10: 32}
        # return {1: 5, 2: 7, 3: 4, 4: 3, 5: 8, 6: 2, 7: 7, 8: 9, 9: 4, 10: 25}
        # return {1: 8, 2: 8, 3: 8, 4: 8, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 24}


    print("=" * 40)
    print("=== TESTING SUPER COMPUTER ===")
    print("=" * 40)

    print(">>> Test 1:")
    deck_1 = get_test_deck()
    get_decision_from_state(deck_1, player_cards=[1, 3], dealer_upcard=10)

    print(">>> Test 2")
    deck_2 = get_test_deck()
    get_decision_from_state(deck_2, player_cards=[1, 4], dealer_upcard=3)

    print(">>> Test 3")
    deck_2 = get_test_deck()
    get_decision_from_state(deck_2, player_cards=[5, 4, 10], dealer_upcard=1)

    print(">>> Test 4")
    deck_2 = get_test_deck()
    get_decision_from_state(deck_2, player_cards=[2, 4, 5], dealer_upcard=2)

    # =========================================================================
    # 3. TEST RUNNER: PRE-ROUND
    # =========================================================================
    print("=" * 40)
    print("=== TESTING PRE-ROUND ===")
    print("=" * 40)

    # Simulate a shoe halfway through, where lots of small cards have been played (High EV)
    rich_deck = get_test_deck()
    rich_deck[2] = 2
    rich_deck[3] = 2
    rich_deck[4] = 2
    rich_deck[5] = 1
    rich_deck[6] = 2

    print(">>> Pre-Round Stats (Rich Deck - Small cards missing):")
    pre_round_stats = evaluate_pre_round_stats(rich_deck, num_decks=2)

    # Print the general stats
    print(f"Expected Value: {pre_round_stats['ev']:+.2f}%")
    print(f"Recommendation: {pre_round_stats['bet_recommendation']}")

    # Print ALL player stats
    p_stats = pre_round_stats['player_stats']
    print(
        f"Player Stats -> Win: {p_stats['win']*100}% | Tie: {p_stats['tie']*100}% | Loss: {p_stats['loss']*100}%")

    # Print ALL dealer stats
    d_stats = pre_round_stats['dealer_probs']
    print(
        f"Dealer Stats -> Bust: {d_stats['Bust']}% | 17: {d_stats[17]}% | 18: {d_stats[18]}% | 19: {d_stats[19]}% | 20: {d_stats[20]}% | 21: {d_stats[21]}%\n")
