import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
from models import Event, EventType, EventSource, Entity, EntityType

class ExtractionPipeline:
    """Extracts, normalizes, deduplicates, and structures entities and events from raw data payloads"""

    def __init__(self):
        self.seen_event_ids = set()
        self.entity_registry: Dict[str, Entity] = {}

    def normalize_entity_name(self, name: str) -> str:
        """Standardizes entity names for entity resolution"""
        cleaned = re.sub(r'[^\w\s-]', '', name).strip()
        return cleaned

    def extract_entities_from_raw(self, raw_item: Dict[str, Any]) -> List[Entity]:
        """Extracts and resolves biopharma entities from incoming payload"""
        extracted = []
        raw_entities = raw_item.get("entities", [])

        for ent in raw_entities:
            name = ent.get("name")
            e_type_raw = ent.get("type", EntityType.ORGANIZATION)
            if not name:
                continue

            # Normalize e_type to EntityType
            if isinstance(e_type_raw, EntityType):
                e_type = e_type_raw
            else:
                try:
                    e_type = EntityType(e_type_raw)
                except ValueError:
                    # Match by enum value case-insensitively or default
                    matched = False
                    for et in EntityType:
                        if et.value.lower() == str(e_type_raw).lower():
                            e_type = et
                            matched = True
                            break
                    if not matched:
                        e_type = EntityType.ORGANIZATION

            norm_name = self.normalize_entity_name(name)
            entity_id = f"{e_type.value.lower()}_{norm_name.lower().replace(' ', '_')}"

            if entity_id in self.entity_registry:
                entity = self.entity_registry[entity_id]
            else:
                entity = Entity(
                    id=entity_id,
                    name=norm_name,
                    entity_type=e_type,
                    synonyms=[name] if name != norm_name else []
                )
                self.entity_registry[entity_id] = entity
            extracted.append(entity)

        return extracted

    def process_raw_item(self, raw_item: Dict[str, Any]) -> Tuple[Event, bool]:
        """Processes raw payload into a validated Event object. Returns (Event, is_new)"""
        event_id = raw_item.get("id", str(uuid.uuid4()))
        is_new = event_id not in self.seen_event_ids
        self.seen_event_ids.add(event_id)

        entities = self.extract_entities_from_raw(raw_item)

        event_type = raw_item.get("event_type", EventType.TRIAL_STATUS_CHANGE)
        source = raw_item.get("source_name", EventSource.NEWS)

        event = Event(
            id=event_id,
            title=raw_item.get("title", "Untitled Event"),
            event_type=event_type,
            source=source,
            timestamp=raw_item.get("timestamp", datetime.utcnow()),
            entities=entities,
            summary=raw_item.get("summary", ""),
            details=raw_item.get("details", ""),
            source_url=raw_item.get("source_url"),
            jurisdiction=raw_item.get("jurisdiction", "Global"),
            historical_version=1
        )
        return event, is_new

    def process_batch(self, raw_items: List[Dict[str, Any]]) -> List[Event]:
        """Processes a batch of raw ingested items into normalized Events"""
        events = []
        for item in raw_items:
            event, is_new = self.process_raw_item(item)
            if is_new:
                events.append(event)
        return events
