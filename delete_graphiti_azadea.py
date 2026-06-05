#!/usr/bin/env python3
"""
Script to delete all Graphiti data for the azadea group from Neo4j.
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GRAPHITI_GROUP_ID = os.getenv("GRAPHITI_GROUP_ID", "azadea")

def delete_graphiti_data():
    """Delete all Graphiti data for the azadea group."""
    print(f"🔗 Connecting to Neo4j: {NEO4J_URI} (database: {NEO4J_DATABASE})")
    print(f"🎯 Target group_id: {GRAPHITI_GROUP_ID}")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            # First, count how many nodes/relationships we'll delete
            print("\n📊 Counting existing data...")
            
            # Count Episodic nodes with this group_id
            count_episodic_query = """
            MATCH (e:Episodic)
            WHERE e.group_id = $group_id
            RETURN count(e) as episodic_count
            """
            result = session.run(count_episodic_query, group_id=GRAPHITI_GROUP_ID)
            episodic_count = result.single()["episodic_count"]
            
            # Count Entity nodes with this group_id
            count_entity_query = """
            MATCH (e:Entity)
            WHERE e.group_id = $group_id
            RETURN count(e) as entity_count
            """
            result = session.run(count_entity_query, group_id=GRAPHITI_GROUP_ID)
            entity_count = result.single()["entity_count"]
            
            # Count relationships
            count_rels_query = """
            MATCH (n {group_id: $group_id})-[r]-()
            RETURN count(r) as rel_count
            """
            result = session.run(count_rels_query, group_id=GRAPHITI_GROUP_ID)
            rel_count = result.single()["rel_count"]
            
            print(f"   Found {episodic_count} Episodic nodes with group_id='{GRAPHITI_GROUP_ID}'")
            print(f"   Found {entity_count} Entity nodes with group_id='{GRAPHITI_GROUP_ID}'")
            print(f"   Found {rel_count} relationships")
            
            total_nodes = episodic_count + entity_count
            if total_nodes == 0:
                print("\n✅ No data found for this group_id. Nothing to delete.")
                return
            
            # Confirm deletion
            print(f"\n⚠️  WARNING: This will delete ALL Graphiti data for group_id='{GRAPHITI_GROUP_ID}'")
            print(f"   This includes {episodic_count} Episodic nodes, {entity_count} Entity nodes, and {rel_count} relationships")
            response = input("   Are you sure you want to proceed? (yes/no): ")
            
            if response.lower() != "yes":
                print("❌ Deletion cancelled.")
                return
            
            print("\n🗑️  Deleting data...")
            
            # Delete all relationships first (to avoid constraint violations)
            delete_relationships_query = """
            MATCH (n {group_id: $group_id})-[r]-()
            DELETE r
            RETURN count(r) as deleted_rels
            """
            result = session.run(delete_relationships_query, group_id=GRAPHITI_GROUP_ID)
            deleted_rels = result.single()["deleted_rels"]
            print(f"   ✅ Deleted {deleted_rels} relationships")
            
            # Delete Entity nodes
            delete_entities_query = """
            MATCH (e:Entity {group_id: $group_id})
            DETACH DELETE e
            RETURN count(e) as deleted_entities
            """
            result = session.run(delete_entities_query, group_id=GRAPHITI_GROUP_ID)
            deleted_entities = result.single()["deleted_entities"]
            print(f"   ✅ Deleted {deleted_entities} Entity nodes")
            
            # Delete Episodic nodes
            delete_episodic_query = """
            MATCH (e:Episodic {group_id: $group_id})
            DETACH DELETE e
            RETURN count(e) as deleted_episodic
            """
            result = session.run(delete_episodic_query, group_id=GRAPHITI_GROUP_ID)
            deleted_episodic = result.single()["deleted_episodic"]
            print(f"   ✅ Deleted {deleted_episodic} Episodic nodes")
            
            # Delete Community nodes (if any)
            delete_community_query = """
            MATCH (c:Community {group_id: $group_id})
            DETACH DELETE c
            RETURN count(c) as deleted_community
            """
            result = session.run(delete_community_query, group_id=GRAPHITI_GROUP_ID)
            deleted_community = result.single()["deleted_community"]
            if deleted_community > 0:
                print(f"   ✅ Deleted {deleted_community} Community nodes")
            
            # Delete any remaining nodes with this group_id (catch-all)
            delete_remaining_query = """
            MATCH (n {group_id: $group_id})
            DETACH DELETE n
            RETURN count(n) as deleted_remaining
            """
            result = session.run(delete_remaining_query, group_id=GRAPHITI_GROUP_ID)
            deleted_remaining = result.single()["deleted_remaining"]
            if deleted_remaining > 0:
                print(f"   ✅ Deleted {deleted_remaining} remaining nodes")
            
            # Verify deletion
            verify_episodic_query = """
            MATCH (e:Episodic {group_id: $group_id})
            RETURN count(e) as remaining_episodic
            """
            result = session.run(verify_episodic_query, group_id=GRAPHITI_GROUP_ID)
            remaining_episodic = result.single()["remaining_episodic"]
            
            verify_entity_query = """
            MATCH (e:Entity {group_id: $group_id})
            RETURN count(e) as remaining_entity
            """
            result = session.run(verify_entity_query, group_id=GRAPHITI_GROUP_ID)
            remaining_entity = result.single()["remaining_entity"]
            
            if remaining_episodic == 0 and remaining_entity == 0:
                print(f"\n✅ Successfully deleted all Graphiti data for group_id='{GRAPHITI_GROUP_ID}'")
                print(f"   Total deleted: {deleted_episodic} Episodic nodes, {deleted_entities} Entity nodes, {deleted_rels} relationships")
            else:
                print(f"\n⚠️  Warning: {remaining_episodic} Episodic and {remaining_entity} Entity nodes still remain. Deletion may be incomplete.")
                
    except Exception as e:
        print(f"\n❌ Error deleting data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        driver.close()
        print("\n🔌 Disconnected from Neo4j")

if __name__ == "__main__":
    delete_graphiti_data()
