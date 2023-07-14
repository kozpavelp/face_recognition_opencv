import asyncio
from recognition.recognition_module import run_recognition


async def main():
    await run_recognition()

if __name__ == '__main__':
    asyncio.run(main())
