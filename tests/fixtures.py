from fcg_ia5.src.algorithm import CATEGORICAL, GROUPS, NUMERIC, Support
from fcg_paper_faithful.src.synthetic import Development


def development(groups=4):
    return tuple(Development(f"synthetic:dev:{group}:{y}", group, y, group * 5 + y)
                 for group in range(groups) for y in (0, 1))


def support():
    return tuple(
        Support(f"synthetic:support:{g}:{y}:{i:02}", sex, race, y,
                dict.fromkeys(NUMERIC, i // 2) | dict.fromkeys(CATEGORICAL, "synthetic-category"))
        for g, (sex, race) in enumerate(GROUPS) for y in (0, 1) for i in range(10)
    )
