"""Room registry and maintenance utilities."""

import random
import string
import time

games: dict = {}

ROOM_CODE_LENGTH = 6
ROOM_EXPIRY_SECONDS = 3600  # 1 hour


def generate_room_code() -> str:
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=ROOM_CODE_LENGTH))
        if code not in games:
            return code


def prune_old_games():
    now = time.time()
    expired = [gid for gid, room in games.items()
               if now - room.last_activity > ROOM_EXPIRY_SECONDS]
    for gid in expired:
        del games[gid]
