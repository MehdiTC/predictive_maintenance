"""Canonical tabular schema for the C-MAPSS raw files.

The raw trajectory files carry no header. Per the dataset's own readme, each
row is one operating cycle of one engine unit: unit number, cycle number,
three operational settings, and twenty-one sensor measurements. The sensor
channels are anonymous; this project deliberately does not assign them
physical interpretations (such as vibration or temperature), so the canonical
names are positional (``sensor_01`` ... ``sensor_21``).

The official RUL file carries one integer per test unit: the remaining useful
life (in cycles) of that unit at the end of its recorded test trajectory.
"""

SCHEMA_VERSION = "1"

ASSET_ID_COLUMN = "asset_id"
CYCLE_COLUMN = "cycle"

OPERATING_SETTING_COLUMNS: tuple[str, ...] = tuple(
    f"operating_setting_{index}" for index in range(1, 4)
)
SENSOR_COLUMNS: tuple[str, ...] = tuple(f"sensor_{index:02d}" for index in range(1, 22))

TRAJECTORY_COLUMNS: tuple[str, ...] = (
    ASSET_ID_COLUMN,
    CYCLE_COLUMN,
    *OPERATING_SETTING_COLUMNS,
    *SENSOR_COLUMNS,
)
"""Canonical trajectory column order; also the raw files' positional order."""

TRAJECTORY_INTEGER_COLUMNS: tuple[str, ...] = (ASSET_ID_COLUMN, CYCLE_COLUMN)
TRAJECTORY_FLOAT_COLUMNS: tuple[str, ...] = (*OPERATING_SETTING_COLUMNS, *SENSOR_COLUMNS)

TRAJECTORY_DTYPES: dict[str, str] = {
    **dict.fromkeys(TRAJECTORY_INTEGER_COLUMNS, "int64"),
    **dict.fromkeys(TRAJECTORY_FLOAT_COLUMNS, "float64"),
}

RUL_COLUMN = "rul"
RUL_COLUMNS: tuple[str, ...] = (RUL_COLUMN,)
RUL_DTYPES: dict[str, str] = {RUL_COLUMN: "int64"}
