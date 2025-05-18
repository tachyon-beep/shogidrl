# Outstanding Fixes & Improvements for Shogi Engine

This document lists identified bugs, logic errors, and areas for improvement in `shogi_engine.py` based on feedback received on May 18, 2025.

## I. General Design Choices & Improvements

1.  **Piece Representation & Promotion Consistency:**
    *   **Issue:** The current system uses both `piece.type` (with types 8-13 for promoted pieces) and a separate `piece.is_promoted` flag. This duality is a potential source of bugs and inconsistencies.
    *   **Recommendation (Best Practice for DRL):** Adopt **Refined Option B**.
        *   `piece.type` will be the single source of truth (e.g., 0-7 for base, 8-13 for promoted).
        *   `piece.is_promoted` will be a read-only `@property` in the `Piece` class, deriving its value from `piece.type`.
    *   **Sub-Tasks for Refined Option B:**
        *   [ ] Modify `Piece` class:
            *   Remove `is_promoted` as a stored attribute in `__init__`. `__init__` should only take `piece_type_int` and `color_int`.
            *   Implement `is_promoted(self) -> bool` as a `@property` that returns `True` if `self.type` corresponds to a known promoted piece type (e.g., types 8-13). Define a clear set of promoted type IDs (e.g., `PROMOTED_TYPES_SET = {8, 9, 10, 11, 12, 13}`).
        *   [ ] Update `make_move` (see II.4): Ensure it changes `moving_piece.type` from its base type to its corresponding promoted type ID when a piece promotes.
        *   [ ] Update `undo_move` (see II.6): Ensure it reverts `moving_piece.type` from a promoted type ID back to its base type ID when a promotion is undone.
        *   [ ] Review and update all functions that currently use `is_promoted` as a stored flag. They should now rely on the `piece.is_promoted` property or directly on `piece.type` where appropriate.
    *   **Impact:** Affects `Piece.symbol()`, `get_observation()`, `make_move()`, `undo_move()`, `add_to_hand()`.

2.  **Use of Constants (Enums):**
    *   **Suggestion:** Replace raw integers for piece types and colors with Python Enums.
    *   **Sub-Tasks:**
        *   [ ] Define `PieceType(Enum)` in `shogi_engine.py` (or a constants file) with members like `PAWN = 0`, `LANCE = 1`, ..., `PROMOTED_PAWN = 8`, ..., `PROMOTED_ROOK = 13` (if Option B for promotion) or just base types (if Option A).
        *   [ ] Define `Color(Enum)` in `shogi_engine.py` (or a constants file) with members `BLACK = 0`, `WHITE = 1`.
        *   [ ] Throughout `shogi_engine.py`, replace integer literals for piece types (e.g., `0`, `1`, `8`) with their Enum equivalents (e.g., `PieceType.PAWN`, `PieceType.LANCE`, `PieceType.PROMOTED_PAWN`).
        *   [ ] Throughout `shogi_engine.py`, replace integer literals for colors (e.g., `0`, `1`) with their Enum equivalents (e.g., `Color.BLACK`, `Color.WHITE`).
        *   [ ] Update type hints to use these Enums where appropriate (e.g., `piece_type: PieceType`, `color: Color`).
    *   **Benefit:** Improves code readability, maintainability, and type safety.

3.  **Clarity of `get_individual_piece_moves()`:**
    *   **Issue:** The comment "Always add the king\'s square as a possible move for check detection" for sliding pieces is non-standard.
    *   **Sub-Tasks:**
        *   [ ] **Analyze Usage:** Determine all call sites of `get_individual_piece_moves()`. Understand if they expect "legal landing squares" or "all attacked squares including through pieces to a king."
        *   [ ] **Refactor (If Necessary):**
            *   If distinct functionalities are needed:
                *   [ ] Rename current `get_individual_piece_moves()` to something like `_get_attacked_squares_for_check_detection()` if its primary use is for that specific king-attack scenario.
                *   [ ] Create a new function, e.g., `_get_standard_piece_moves(piece, r_from, c_from)`, that generates only valid landing squares (empty or opponent, stopping at first piece).
            *   If the current function can be clarified:
                *   [ ] Update the docstring and comments within `get_individual_piece_moves()` to precisely define its behavior regarding sliding pieces and king attacks. Explain *why* it includes squares beyond the first blocking piece for check detection if that\'s intended.
        *   [ ] Ensure callers use the correct function for their needs.

4.  **`is_on_board()` `None` Check for Coordinates:**
    *   **Issue:** The check `if row is None or col is None: return False` is unusual.
    *   **Sub-Tasks:**
        *   [ ] **Trace Callers:** Identify how `row` or `col` could become `None` when `is_on_board()` is called. Look at move generation logic in `get_legal_moves()` and `_get_individual_piece_moves()`.
        *   [ ] **Fix Upstream (If `None` is possible and incorrect):** If `None` values are found to be passed due to a bug elsewhere, fix that upstream bug.
        *   [ ] **Evaluate Necessity:**
            *   If `None` values are never legitimately passed, remove the `if row is None or col is None:` check.
            *   If `None` values *can* be legitimately passed by design (though unlikely for board coordinates), keep the check but document why `None` is a possible input.
            *   Consider replacing with `assert row is not None and col is not None` if these should never be `None`, to catch errors during development.

## II. Specific Logic Bugs and Errors

1.  **`Piece.symbol()` Incorrect for Promoted Base Types:**
    *   **Bug:** If `Piece.type` is a base type (e.g., 0 for Pawn) and `Piece.is_promoted` is `True`, `symbol()` returns "P", not "+P".
    *   **Sub-Tasks (Depends on I.1 decision):**
        *   **If Promotion Option A (base type + `is_promoted` flag):**
            *   [ ] Modify `Piece.symbol()`:
                *   Get the base symbol using `self.type` (e.g., "P" for Pawn).
                *   If `self.is_promoted` is `True` AND the piece type is promotable (e.g., not Gold or King), prepend "+" to the symbol.
        *   **If Promotion Option B (hybrid `type` 8-13 and `is_promoted`):**
            *   [ ] Ensure `make_move` correctly updates `piece.type` to its promoted version (e.g., 0 to 8) when a piece is promoted (see II.4).
            *   [ ] `Piece.symbol()` can then primarily rely on `self.type` to fetch the correct symbol from `base_symbols` (which should include entries for promoted types like `8: "+P"`). The `is_promoted` flag would be redundant for symbol generation if `type` is canonical.
    *   **Fix:** Align with the chosen canonical promotion representation (see I.1).

2.  **`get_observation()` Incorrect Channel Mapping for Promoted Pieces:**
    *   **Bug:** `ch = base + 7 + (t % 7)` incorrectly maps promoted piece types to observation channels.
    *   **Sub-Tasks:**
        *   [ ] **Define Correct Mapping:** Create a clear mapping from a piece\'s *base type* to its corresponding channel index within the "promoted pieces" block of the observation tensor.
            *   Example: Pawn (type 0) promotes to +Pawn, which should map to the 0th channel in the promoted block (i.e., overall channel `base_offset_for_player + 7 + 0`).
            *   Lance (type 1) promotes to +Lance, maps to 1st channel in promoted block (overall channel `base_offset_for_player + 7 + 1`).
            *   ...
            *   Bishop (type 5) promotes to +Bishop, maps to 4th channel in promoted block.
            *   Rook (type 6) promotes to +Rook, maps to 5th channel in promoted block.
            *   (Gold and King do not have separate "promoted" channels in this scheme).
        *   [ ] **Modify `get_observation()` Logic:**
            *   Inside the loop iterating through board pieces:
                *   Determine the piece\'s base type (e.g., if `p.type` is 8 (+Pawn), its base type is 0 (Pawn)). This might require a helper or a small map if using Option B for promotion. If Option A, `p.type` is already the base type.
                *   Let this be `base_piece_type_for_channel`.
                *   Determine if the piece is actually promoted (either `p.is_promoted` is `True`, or `p.type` is 8-13 if using Option B).
                *   If promoted AND `base_piece_type_for_channel` is one that has a distinct promoted plane (P, L, N, S, B, R):
                    *   Use the mapping from the previous step to find `promoted_plane_idx` (0-5).
                    *   Set `ch = base_offset_for_player + 7 + promoted_plane_idx`.
                *   Else (if unpromoted, or Gold/King):
                    *   Set `ch = base_offset_for_player + base_piece_type_for_channel`.
                *   Set `obs[ch, r, c] = 1.0`.
        *   [ ] **Verify Channel Definitions:** Ensure the total number of channels and their assignments match the design document (7 unpromoted planes + 6 promoted planes per player, plus other state planes).
    *   **Reference:** Design doc: 7 unpromoted (0-6: P,L,N,S,G,B,R), 6 promoted (7-12: +P,+L,+N,+S,+B,+R).

3.  **`is_uchi_fu_zume()` Path Clearing for Opponent\'s Promoted Sliding Pieces:**
    *   **Bug:** Logic `if piece.type in [1, 5, 6]:` only checks unpromoted L, B, R for path clearing, missing +B (type 12) and +R (type 13).
    *   **Sub-Tasks:**
        *   [ ] **Identify Sliding Pieces Correctly:**
            *   Modify the condition to include promoted sliding pieces: `if piece.type in [1, 5, 6, 12, 13]:`
            *   Alternatively, create a helper function `_is_sliding_piece(piece_type: int) -> bool` that returns `True` if the piece type (promoted or unpromoted) is a sliding piece (Lance, Bishop, Rook, +Bishop, +Rook). Use this helper in the condition.
        *   [ ] **Verify Path Checking Logic:** Ensure the existing path checking code within this `if` block correctly determines the direction and extent of slides for *both* unpromoted and promoted sliders. (Promoted Bishop/Rook have the same sliding directions as their unpromoted counterparts).

4.  **`make_move()` Must Update `piece.type` on Promotion (If Using Hybrid Model - Option B of I.1):**
    *   **Omission/Bug:** If using `piece.type` 8-13 to denote promoted pieces, `make_move` only sets `moving_piece.is_promoted = True` and does not change `moving_piece.type`.
    *   **Sub-Tasks (Only if Promotion Option B from I.1 is chosen):**
        *   [ ] **Define Promotion Type Map:** Create a dictionary mapping base piece types to their promoted types, e.g., `BASE_TO_PROMOTED_TYPE = {0:8, 1:9, 2:10, 3:11, 5:12, 6:13}`.
        *   [ ] **Modify `make_move()`:**
            *   When `promote is True` and `moving_piece` is not `None` and `not moving_piece.is_promoted`:
                *   Set `moving_piece.is_promoted = True`.
                *   If `moving_piece.type` is in `BASE_TO_PROMOTED_TYPE`:
                    *   Store `original_base_type = moving_piece.type` in the `move_details` dictionary for `self.move_history` (e.g., `move_details["original_type_before_promotion"] = original_base_type`). This is crucial for `undo_move`.
                    *   Update `moving_piece.type = BASE_TO_PROMOTED_TYPE[moving_piece.type]`.
        *   [ ] **Test:** Ensure promotion correctly changes both attributes.

5.  **`undo_move()` Incorrect Hand Restoration for Drops:**
    *   **Bug:** Piece returned to `self.hands[self.current_player]` instead of the player who made the drop.
    *   **Sub-Tasks:**
        *   [ ] **Locate Code:** In `undo_move()`, find the section handling `if last_move_type == "drop":`.
        *   [ ] **Correct Player Index:** Change `self.hands[self.current_player][piece_type] += 1` to `self.hands[1 - self.current_player][piece_type] += 1`.
        *   **Explanation:** `self.current_player` at this point in `undo_move` has already been toggled to the player whose turn it *was*. The piece belongs to the *other* player (who made the move being undone).

6.  **`undo_move()` Must Revert `piece.type` on Demotion (If Using Hybrid Model - Option B of I.1):**
    *   **Omission/Bug:** If `make_move` changes `piece.type` on promotion, `undo_move` only sets `moving_piece.is_promoted = False` and doesn\'t revert `piece.type`.
    *   **Sub-Tasks (Only if Promotion Option B from I.1 is chosen AND II.4 is implemented):**
        *   [ ] **Define Demotion Type Map (or use inverse of II.4 map):** Create `PROMOTED_TO_BASE_TYPE = {8:0, 9:1, 10:2, 11:3, 12:5, 13:6}`.
        *   [ ] **Modify `undo_move()`:**
            *   In the section handling board moves (not drops), when `last.get("was_promoted", False)` is true and `moving_piece` exists and `moving_piece.is_promoted` is true (before setting it to False):
                *   Set `moving_piece.is_promoted = False`.
                *   If `moving_piece.type` is in `PROMOTED_TO_BASE_TYPE`:
                    *   Update `moving_piece.type = PROMOTED_TO_BASE_TYPE[moving_piece.type]`.
                *   Alternatively, if `move_details["original_type_before_promotion"]` was stored (from II.4), use that: `moving_piece.type = last["original_type_before_promotion"]`. This is more robust if a piece could theoretically be promoted then captured and its type changed in hand (though not standard Shogi). Using the stored original type is safer.
        *   [ ] **Test:** Ensure demotion correctly reverts both attributes.

7.  **`add_to_hand()` Incorrect Base Type Calculation for Captured Promoted Pieces:**
    *   **Bug:** `base_type = piece.type % 7` incorrectly converts captured promoted pieces (e.g., +Pawn type 8 becomes Lance type 1).
    *   **Sub-Tasks:**
        *   [ ] **Define Correct Mapping:** Create a dictionary `PIECE_TYPE_TO_HAND_TYPE = { ... }` that maps *any* piece type (0-13, or just 0-7 if Option A for promotion) to its fundamental, unpromoted type suitable for being held in hand (0-6, as Kings type 7 are not held).
            *   `0:0` (Pawn -> Pawn)
            *   `1:1` (Lance -> Lance) ...
            *   `6:6` (Rook -> Rook)
            *   `8:0` (+Pawn -> Pawn)
            *   `9:1` (+Lance -> Lance) ...
            *   `11:3` (+Silver -> Silver)
            *   `12:5` (+Bishop -> Bishop)
            *   `13:6` (+Rook -> Rook)
            *   (Gold type 4 maps to 4. King type 7 should be excluded before this map or return a special value indicating it cannot be added to hand).
        *   [ ] **Modify `add_to_hand()`:**
            *   Check if `piece.type == 7` (King). If so, `return` immediately as Kings cannot be captured or held.
            *   Use `base_type = PIECE_TYPE_TO_HAND_TYPE.get(piece.type)`.
            *   If `base_type` is not `None` (i.e., the piece type was valid and mappable):
                *   Proceed with `self.hands[color][base_type] += 1`.
            *   Else (if `piece.type` was invalid or unmapped, though this shouldn\'t happen with valid pieces):
                *   Consider logging an error or raising an exception.
        *   [ ] **Test:** Verify that capturing any piece (promoted or unpromoted) adds the correct *unpromoted base piece* to the capturing player's hand.

8.  **Sennichite (Fourfold Repetition) Off-by-One Error:**
    *   **Issue:** The current check for Sennichite (`state_counts[current_state_hash] >= 3`) might trigger on the third appearance of a state instead of the required fourth, because the current state is added to history *before* the check.
    *   **Confidence:** Moderate.
    *   **Sub-Tasks:**
        *   [ ] Locate the `is_sennichite` method (or equivalent logic) and where it's called (likely after a move is made).
        *   [ ] Verify the order of operations: when the current board state hash is added to `self.state_history` (or `state_counts` is updated) versus when the Sennichite condition is checked.
        *   [ ] **If state is added *before* check:** Modify the condition to `state_counts[current_state_hash] >= 4` to correctly identify the fourth occurrence.
        *   [ ] **Alternatively, if preferred:** Modify logic to call `is_sennichite` (or perform the check) *before* the current state is recorded in the history for the current move. In this case, the check would remain `state_counts[current_state_hash] >= 3` (as it would be checking against three *previous* identical states). Ensure comments clarify the chosen logic.
        *   [ ] Add unit tests specifically for Sennichite detection, covering cases that reach the exact threshold.

9.  **Drop Simulation in `get_legal_moves` Does Not Temporarily Debit Hand:**
    *   **Issue:** When `get_legal_moves` simulates a drop to check for legality (e.g., self-check), it places the piece on the board but does not temporarily decrement the piece count from the player's hand during the simulation.
    *   **Confidence:** Even chance (minor issue, potential for problems with very complex future rule checks).
    *   **Sub-Tasks:**
        *   [ ] In `get_legal_moves`, identify the section where drop moves are generated and validated.
        *   [ ] When a drop is simulated (piece placed on board temporarily):
            *   [ ] Before `set_piece()` to place the dropped piece, temporarily decrement the count of that `piece_type` in `self.hands[self.current_player]`.
            *   [ ] After the check (e.g., `is_check` on the temporary board) and before or after `set_piece(r, c, None)` to remove the simulated piece, restore the hand count by incrementing it back.
        *   [ ] Ensure this temporary modification of hand count is correctly undone for all paths (e.g., even if the drop is illegal).
        *   [ ] Evaluate if this change is critical for current rules or primarily for robustness against hypothetical future rule interactions.

10. **Implement Rare Promotion Rule (Decline only if future moves exist from destination):**
    *   **Issue:** The engine may not correctly implement the nuanced Shogi rule that a player can only decline an optional promotion if the piece, in its unpromoted state, would still have legal moves from the destination square. If declining promotion would leave the piece stranded with no moves, promotion becomes mandatory.
    *   **Confidence:** Low (edge case rule).
    *   **Sub-Tasks:**
        *   [ ] **Clarify Rule:** Confirm the exact interpretation of this rule as implemented in standard Shogi engines or rule sets. (The most common interpretation is: if a piece *can* promote, and it's not a mandatory promotion like a pawn on the last rank, the player can choose not to promote. However, if *not* promoting would leave that piece with *zero legal moves* from its destination square on its *subsequent turn*, then promotion is forced).
        *   [ ] **Locate Promotion Logic:** Identify where promotion options are generated in `get_legal_moves` (e.g., when `(r_to, c_to, True)` moves are added) and/or handled in `make_move`.
        *   [ ] **Modify Logic:**
            *   When a piece makes a move into the promotion zone and promotion is not already mandatory (e.g., pawn/lance on final rank, knight on final two ranks):
                *   The system should generate two potential moves: one with promotion, one without (if optional).
                *   For the move *without* promotion:
                    *   Perform a lookahead: temporarily place the unpromoted piece on the destination square.
                    *   Check if this unpromoted piece has any legal moves from that destination square (e.g., by calling a lightweight version of `get_legal_moves` for just that piece).
                    *   If the unpromoted piece has *no* legal moves from the destination square, then the option to *not* promote is invalid. In this scenario, only the promoted move should be considered legal (or the promotion flag should be forced if the player attempts to move without promoting).
        *   [ ] Add specific unit tests for scenarios involving this rule, particularly for pieces like Knights, Lances, and Silvers near the promotion zone.

11. **Incorrect Move Generation for King Attacks by Sliding Pieces:**
    *   **Issue:** The logic for generating moves/attacks for sliding pieces (Lance, Rook, Bishop) might incorrectly list the square of one's *own* king as a valid target or attacked square.
    *   **Confidence:** Moderate (based on feedback).
    *   **Sub-Tasks:**
        *   [ ] Review the move generation loop within `_get_individual_piece_moves` (or any equivalent function) for sliding pieces (Lances, Rooks, Bishops, and their promoted forms).
        *   [ ] When a sliding piece's potential path encounters another piece (`target = self.board[nr][nc]`):
            *   If `target` is a King (`target.type == KING_TYPE_ID` or `PieceType.KING`):
                *   Ensure that the move/attack is only considered valid (and `(nr, nc)` is added to `moves`) if `target.color != piece.color` (i.e., it's the opponent's King).
                *   If `target.color == piece.color` (it's one's own King), the path should be blocked for further moves in that direction, and `(nr, nc)` should *not* be added as a move.
            *   If `target` is not a King but `target.color == piece.color` (another friendly piece), the path is blocked.
            *   If `target.color != piece.color` (opponent's piece, not King), `(nr, nc)` is a valid move/capture, and the path is blocked.
        *   [ ] Add unit tests that specifically place a friendly King in the path of a friendly sliding piece to ensure it's not listed as a valid move or attack.

12. **Initial Board Setup Verification (Especially Bishop/Rook):**
    *   **Issue:** The initial placement of pieces, particularly Black's Bishop and Rook, might not conform to the standard Shogi starting position.
    *   **Confidence:** Moderate (worth verifying).
    *   **Standard Setup (0-indexed [row][col], Black at bottom rows 0-2, White at top rows 6-8):**
        *   Black's King: `(0, 4)`
        *   Black's Rook: `(1, 1)` (Standard: 2h)
        *   Black's Bishop: `(1, 7)` (Standard: 8h)
        *   White's King: `(8, 4)`
        *   White's Rook: `(7, 7)` (Standard: 8b)
        *   White's Bishop: `(7, 1)` (Standard: 2b)
    *   **Sub-Tasks:**
        *   [ ] Locate the `__init__` method of the `ShogiGame` class (or wherever the board is initialized).
        *   [ ] Compare the programmed initial positions of all pieces, especially the Rooks and Bishops for both Black and White, against the standard Shogi setup.
            *   (Example for Black's Rook, assuming `Piece(PIECE_TYPE_ENUM.ROOK, COLOR_ENUM.BLACK)`): `self.board[1][1]` should be this piece.
            *   (Example for Black's Bishop): `self.board[1][7]` should be this piece.
        *   [ ] Correct any discrepancies found in the initial setup.
        *   [ ] Create or enhance a unit test (e.g., `test_initial_board_setup`) that asserts the correct type and color of pieces at all 81 squares for the standard starting position.

13. **Mandatory Promotion Logic Verification:**
    *   **Issue:** The engine must correctly enforce mandatory promotions for Pawns, Lances, and Knights when they reach ranks where they have no further unpromoted moves.
    *   **Confidence:** High (critical rule).
    *   **Mandatory Promotion Conditions (0-indexed rows, Black moves from row 0 upwards, White from row 8 downwards):**
        *   Pawn (Black): Reaches row 8 (opponent's last rank).
        *   Lance (Black): Reaches row 8.
        *   Knight (Black): Reaches row 7 or 8 (opponent's last two ranks).
        *   Pawn (White): Reaches row 0.
        *   Lance (White): Reaches row 0.
        *   Knight (White): Reaches row 1 or 0.
    *   **Sub-Tasks:**
        *   [ ] Review `get_legal_moves`: When generating moves for P, L, N:
            *   If a move takes the piece to a mandatory promotion square (see conditions above):
                *   Ensure only the promoted version of the move `(r_from, c_from, r_to, c_to, True)` is generated.
                *   The non-promoting version `(..., False)` should NOT be generated.
        *   [ ] Review `make_move`:
            *   If a move `(r_from, c_from, r_to, c_to, promote_flag)` is passed:
                *   Check if the move `(r_from, c_from, r_to, c_to)` for the given piece type results in a mandatory promotion condition.
                *   If it is a mandatory promotion condition, the `promote_flag` effectively *must* be true. If `promote_flag` is `False` in such a case, either `make_move` should override it to `True`, or `get_legal_moves` should have prevented such a move from being considered legal. (The latter is preferred: `get_legal_moves` should only offer the mandatory promotion).
        *   [ ] Add unit tests for each piece type (Pawn, Lance, Knight) for both colors, moving to each of their mandatory promotion squares, ensuring only promotion occurs and no non-promoted move is possible or accepted.

---

## III. Performance, Robustness, and Minor Refinements

This section covers issues related to performance optimizations, code robustness for future development (e.g., multithreading), and minor logical refinements that may not be critical bugs but improve code quality or adherence to subtle game rules.

1.  **Promoted Bishop (+B) Movement Generation Redundancy:**
    *   **Issue:** The movement generation for Promoted Bishop (`+B`) might be using the full `king_move_offsets` for its king-like steps, which includes diagonal moves. Since a Promoted Bishop already has its primary diagonal sliding moves, re-adding these via king offsets is redundant.
    *   **Confidence:** Low (minor performance/cleanup).
    *   **Sub-Tasks:**
        *   [ ] Review the part of `_get_individual_piece_moves` (or related functions) that calculates moves for a Promoted Bishop.
        *   [ ] Identify if the king-like component of its movement uses `king_move_offsets` (which includes 4 diagonal and 4 orthogonal steps).
        *   [ ] If so, modify the logic for the Promoted Bishop's king-like steps to *only* add the orthogonal (straight) king steps. The diagonal sliding capabilities should be handled by its bishop-like component.
        *   [ ] Ensure this change does not alter the set of legally generated moves (duplicates are typically filtered out later, but this avoids generating them in the first place).

2.  **Shallow Board Copies in `is_uchi_fu_zume` Simulation:**
    *   **Issue:** The `is_uchi_fu_zume` function, when simulating king escapes, might be using shallow copies of the board or directly manipulating `Piece` objects from the main board state. While the state is restored, this could lead to issues if the engine were ever multithreaded, as another thread might observe inconsistent piece states.
    *   **Confidence:** Low (future-proofing for potential multithreading).
    *   **Sub-Tasks:**
        *   [ ] Examine the board manipulation logic within `is_uchi_fu_zume` during its simulation of opponent king moves.
        *   [ ] Determine if `Piece` objects on the board are being moved by reference.
        *   [ ] **If shallow copies/direct manipulation is confirmed:**
            *   [ ] Option 1 (Preferred for safety): Modify the simulation to operate on a temporary, independent representation of the board (e.g., a 2D array of piece types and colors, or deep-copied `Piece` objects for the relevant parts of the board).
            *   [ ] Option 2 (If full deep copy is too slow): Ensure that any `Piece` object whose state is temporarily changed (e.g., its `r`, `c` attributes) is a *copy* of the original piece, not the original piece itself, if those attributes are part of the `Piece` object.
        *   [ ] Verify that the original board state remains untouched by the simulation and is correctly represented after `is_uchi_fu_zume` completes.
        *   [ ] Note: This is primarily a concern for future architectural changes like multithreading.

3.  **Potential Redundant Path Checking in `_is_square_attacked`:**
    *   **Issue:** The `_is_square_attacked` function might be performing its own path checking for sliding pieces, even if the upstream functions (like `_get_individual_piece_moves` which provides the set of attacked squares) have already filtered moves/attacks based on pieces blocking the path. This could be redundant work.
    *   **Confidence:** Low (performance nit).
    *   **Sub-Tasks:**
        *   [ ] Analyze how `_is_square_attacked` is used, particularly in contexts like check detection.
        *   [ ] Examine the logic within `_is_square_attacked` for sliding pieces (Lance, Bishop, Rook, and their promoted forms).
        *   [ ] Compare this with the logic in `_get_individual_piece_moves` (or equivalent) that generates the attack lines for these sliders.
        *   [ ] If `_get_individual_piece_moves` already ensures that its output (squares attacked by a slider) correctly accounts for blocking pieces (i.e., it doesn't list squares beyond the first obstruction), then `_is_square_attacked` might not need to re-verify the path to the target square for these sliders.
        *   [ ] If redundancy is found, simplify `_is_square_attacked` for sliders, potentially by trusting that if a square is in the attacker_info passed to it, the path is clear. Or, ensure `_get_individual_piece_moves` is the sole place pathing is computed.
