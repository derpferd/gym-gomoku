from gym_gomoku import Board, Color, GomokuState


def test_hash_1():
    b = Board(3)
    b = b.play(3, Color.black)
    b = b.play(7, Color.white)

    correct_strings = [
        """
        000
        100
        020""",
        """
        020
        100
        000""",
        """
        000
        001
        020""",
        """
        000
        200
        010""",
        """
        010
        002
        000""",
    ]

    hashes = list(sorted(map(lambda x: int(x, 3), b.ternary)))
    correct_hashes = list(sorted(int("".join(filter(lambda x: x in "012", s)), 3) for s in correct_strings))
    assert hashes == correct_hashes


def test_state_hash():
    b = Board(3)
    b = b.play(3, Color.black)
    b = b.play(7, Color.white)
    assert hash(GomokuState(b, Color.black)) == 33
    assert hash(GomokuState(b, Color.white)) == 19716

