from sqlmodel import Session, select
from .models import Scan


def create_scan(session: Session, scan: Scan) -> Scan:
    session.add(scan)
    session.commit()
    session.refresh(scan)
    return scan


def get_scans(session: Session, skip: int = 0, limit: int = 100):
    statement = select(Scan).offset(skip).limit(limit)
    return session.exec(statement).all()


def get_scan_by_id(session: Session, scan_id: int) -> Scan | None:
    return session.get(Scan, scan_id)


def delete_scan(session: Session, scan_id: int) -> bool:
    scan = session.get(Scan, scan_id)
    if not scan:
        return False
    session.delete(scan)
    session.commit()
    return True
