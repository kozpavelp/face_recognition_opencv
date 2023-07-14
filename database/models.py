from sqlalchemy import Integer, Column, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class Employees(Base):
    __tablename__ = 'employees'

    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)


class Events(Base):
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True)
    event_time = Column(DateTime, nullable=False)
    photo_path = Column(String(255), nullable=False)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    employee = relationship('Employees')

async def create_tables():
    async with async_session() as session:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)