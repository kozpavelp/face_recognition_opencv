import asyncio

from db_connection import async_session, engine
from models import Events, Employees, Base

events = Events()
employees = Employees()

async def create_tables():
    async with async_session() as session:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

asyncio.run(create_tables())