"""API routes and page serving."""

from flask import Blueprint, jsonify, request, send_from_directory

from ai import pick_ai_move
from models import GameRoom, generate_token
from room_manager import games, generate_room_code, prune_old_games

api_bp = Blueprint("api", __name__)


def get_room_and_player() -> tuple:
    """Extract game_id and token from request, return (room, player_num, error)."""
    game_id = request.args.get("game_id") or (request.get_json(silent=True) or {}).get("game_id")
    token = request.args.get("token") or (request.get_json(silent=True) or {}).get("token")
    if not game_id or not token:
        return None, None, "Missing game_id or token"
    room = games.get(game_id)
    if not room:
        return None, None, "Game not found"
    player = room.player_for_token(token)
    if player is None:
        return None, None, "Invalid token"
    room.touch()

    # Computer mode: human is always player 1
    if room.mode == "computer":
        player = 1
        return room, player, None

    # Local mode: single token shared by both players.
    # Use viewer query param if present, otherwise infer from game state.
    if room.mode == "local":
        viewer = request.args.get("viewer")
        if viewer is not None:
            try:
                player = int(viewer)
            except ValueError:
                pass
        elif room.game.phase == "setup":
            player = room.game.setup_player
        elif room.game.phase in ("play", "over"):
            player = room.game.current_player

    return room, player, None


# ── Page routes ──────────────────────────────────────────────────────────────


@api_bp.route("/")
def index():
    return send_from_directory("static", "menu.html")


@api_bp.route("/local")
def local_page():
    return send_from_directory("static", "local.html")


@api_bp.route("/computer")
def computer_page():
    return send_from_directory("static", "computer.html")


@api_bp.route("/online")
def online_page():
    return send_from_directory("static", "online.html")


# ── Room management ─────────────────────────────────────────────────────────


@api_bp.route("/api/rooms/create", methods=["POST"])
def create_room():
    prune_old_games()
    data = request.get_json(force=True)
    size = data.get("size", 8)
    mode = data.get("mode", "local")
    if size < 4 or size > 16:
        return jsonify({"error": "Board size must be between 4 and 16"}), 400
    if mode not in ("local", "online", "computer"):
        return jsonify({"error": "Invalid mode"}), 400

    game_id = generate_room_code()
    room = GameRoom(game_id, size, mode)
    games[game_id] = room

    return jsonify({
        "gameId": game_id,
        "token": room.host_token,
        "player": 1,
        "size": size,
        "distribution": room.game.distribution,
    })


@api_bp.route("/api/rooms/join", methods=["POST"])
def join_room():
    data = request.get_json(force=True)
    game_id = data.get("gameId", "").strip().upper()
    room = games.get(game_id)
    if not room:
        return jsonify({"error": "Room not found"}), 404
    if room.mode != "online":
        return jsonify({"error": "Cannot join a local game"}), 400
    if room.guest_token is not None:
        return jsonify({"error": "Room is full"}), 400

    room.guest_token = generate_token()
    room.player_tokens[room.guest_token] = 2
    room.bump_version()

    return jsonify({
        "gameId": game_id,
        "token": room.guest_token,
        "player": 2,
        "size": room.game.size,
    })


# ── Game state ──────────────────────────────────────────────────────────────


@api_bp.route("/api/state")
def get_state():
    room, player, err = get_room_and_player()
    if err:
        return jsonify({"error": err}), 400

    # Efficient polling: if client sends version and it hasn't changed, return minimal response
    client_version = request.args.get("version")
    if client_version is not None:
        try:
            if int(client_version) == room.version:
                return jsonify({"changed": False, "version": room.version})
        except ValueError:
            pass

    data = room.game.to_dict(player)
    data["version"] = room.version
    data["changed"] = True
    # For online mode, include whether opponent has joined
    if room.mode == "online":
        data["opponentJoined"] = room.guest_token is not None
    return jsonify(data)


# ── Setup ───────────────────────────────────────────────────────────────────


@api_bp.route("/api/setup/place", methods=["POST"])
def setup_place():
    room, player, err = get_room_and_player()
    if err:
        return jsonify({"error": err}), 400
    data = request.get_json(force=True)
    result = room.game.place_setup_piece(data["r"], data["c"], data["type"], player)
    if "error" in result:
        return jsonify(result), 400
    room.bump_version()
    return jsonify(result)


@api_bp.route("/api/setup/remove", methods=["POST"])
def setup_remove():
    room, player, err = get_room_and_player()
    if err:
        return jsonify({"error": err}), 400
    data = request.get_json(force=True)
    result = room.game.remove_setup_piece(data["r"], data["c"], player)
    if "error" in result:
        return jsonify(result), 400
    room.bump_version()
    return jsonify(result)


@api_bp.route("/api/setup/done", methods=["POST"])
def setup_done():
    room, player, err = get_room_and_player()
    if err:
        return jsonify({"error": err}), 400
    result = room.game.finish_setup(player)
    if "error" in result:
        return jsonify(result), 400
    room.bump_version()
    return jsonify(result)


# ── Moves ───────────────────────────────────────────────────────────────────


@api_bp.route("/api/moves", methods=["GET"])
def get_moves():
    room, player, err = get_room_and_player()
    if err:
        return jsonify({"error": err}), 400
    r = int(request.args["r"])
    c = int(request.args["c"])
    return jsonify({"moves": room.game.get_valid_moves(r, c)})


@api_bp.route("/api/move", methods=["POST"])
def make_move():
    room, player, err = get_room_and_player()
    if err:
        return jsonify({"error": err}), 400

    # Validate it's this player's turn
    if room.game.current_player != player:
        return jsonify({"error": "Not your turn"}), 400

    data = request.get_json(force=True)
    result = room.game.make_move(data["fr"], data["fc"], data["tr"], data["tc"])
    if "error" in result:
        return jsonify(result), 400
    room.bump_version()

    # Computer mode: auto-execute AI move after human's move
    if room.mode == "computer" and room.game.phase == "play" and room.game.current_player == 2:
        ai_move = pick_ai_move(room.game, 2)
        if ai_move:
            ai_result = room.game.make_move(ai_move["fr"], ai_move["fc"], ai_move["tr"], ai_move["tc"])
            room.bump_version()
            result["ai_move"] = {
                "fr": ai_move["fr"],
                "fc": ai_move["fc"],
                "tr": ai_move["tr"],
                "tc": ai_move["tc"],
                **ai_result,
            }

    return jsonify(result)
