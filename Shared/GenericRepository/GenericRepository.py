from typing import Any, Dict, List
from motor.motor_asyncio import AsyncIOMotorCollection


class GenericRepository:
    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    async def get_all(self) -> List[Dict[str, Any]]:
        return await self.collection.find().to_list(None)

    async def get_by_id(self, id: str) -> Dict[str, Any] | None:
        return await self.collection.find_one({"_id": id})

    async def create(self, data: Dict[str, Any]) -> Any:
        return await self.collection.insert_one(data)

    async def update(self, id: str, data: Dict[str, Any]) -> Any:
        return await self.collection.update_one({"_id": id}, {"$set": data})

    async def delete(self, id: str) -> Any:
        return await self.collection.delete_one({"_id": id})

    async def delete_all(self) -> Any:
        return await self.collection.delete_many({})

    async def create_many(self, data: List[Dict[str, Any]]) -> Any:
        return await self.collection.insert_many(data)
