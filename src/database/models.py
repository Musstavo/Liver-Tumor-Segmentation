from datetime import datetime, timezone
from sqlmodel import SQLModel, Field


class Scan(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    patient_id: str = Field(index=True)
    filename: str
    upload_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    liver_volume_cm3: float | None = None
    tumor_volume_cm3: float | None = None
    tumor_percentage: float | None = None
    status: str

    procedure: str | None = None
