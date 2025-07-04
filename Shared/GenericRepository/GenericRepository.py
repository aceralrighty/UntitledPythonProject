from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
from motor.motor_asyncio import AsyncIOMotorCollection

T = TypeVar('T')


class GenericRepository(Generic[T]):
    def __init__(
            self,
            collection: AsyncIOMotorCollection,
            serializer: Callable[[Dict[str, Any]], T],
            deserializer: Callable[[T], Dict[str, Any]]
    ):
        self.collection = collection
        self.serializer = serializer  # Convert dict to domain object
        self.deserializer = deserializer  # Convert domain object to dict

    async def get_all(self) -> List[T]:
        """Get all documents from the collection"""
        docs = await self.collection.find().to_list(None)
        return [self.serializer(doc) for doc in docs]

    async def get_by_id(self, id: str) -> Optional[T]:
        """Get a document by ID"""
        doc = await self.collection.find_one({"_id": id})
        return self.serializer(doc) if doc else None

    async def create(self, entity: T) -> T:
        """Create a new document"""
        doc = self.deserializer(entity)
        result = await self.collection.insert_one(doc)
        doc["_id"] = str(result.inserted_id)  # Convert ObjectId to string
        return self.serializer(doc)

    async def update(self, id: str, entity: T) -> Optional[T]:
        """Update an existing document"""
        doc = self.deserializer(entity)
        result = await self.collection.update_one({"_id": id}, {"$set": doc})

        if result.matched_count == 0:
            return None

        updated_doc = await self.collection.find_one({"_id": id})
        return self.serializer(updated_doc) if updated_doc else None

    async def delete(self, id: str) -> bool:
        """Delete a document by ID"""
        result = await self.collection.delete_one({"_id": id})
        return result.deleted_count > 0

    async def delete_all(self) -> int:
        """Delete all documents from the collection"""
        result = await self.collection.delete_many({})
        return result.deleted_count

    async def create_many(self, entities: List[T]) -> List[T]:
        """Create multiple documents"""
        docs = [self.deserializer(entity) for entity in entities]
        result = await self.collection.insert_many(docs)

        # Update the docs with their new IDs
        for doc, inserted_id in zip(docs, result.inserted_ids):
            doc["_id"] = str(inserted_id)

        return [self.serializer(doc) for doc in docs]

    async def find_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        """Find documents by custom criteria"""
        docs = await self.collection.find(criteria).to_list(None)
        return [self.serializer(doc) for doc in docs]

    async def find_one_by_criteria(self, criteria: Dict[str, Any]) -> Optional[T]:
        """Find a single document by criteria"""
        doc = await self.collection.find_one(criteria)
        return self.serializer(doc) if doc else None

    async def count(self, criteria: Dict[str, Any] = None) -> int:
        """Count documents matching criteria"""
        if criteria is None:
            criteria = {}
        return await self.collection.count_documents(criteria)

    async def exists(self, criteria: Dict[str, Any]) -> bool:
        """Check if a document exists matching criteria"""
        count = await self.collection.count_documents(criteria, limit=1)
        return count > 0