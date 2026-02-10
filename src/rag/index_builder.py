"""
RAG index builder using LlamaIndex and Qdrant.
Handles document loading, embedding, and vector index creation.
"""
from pathlib import Path
from typing import List, Optional
import os

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import Config


class IndexBuilder:
    """Builds and persists vector index for maintenance manuals."""
    
    def __init__(
        self,
        manuals_dir: Path = None,
        collection_name: str = None
    ):
        """
        Initialize index builder.
        
        Args:
            manuals_dir: Directory containing manual PDFs/texts
            collection_name: Qdrant collection name
        """
        self.manuals_dir = manuals_dir or Config.MANUALS_DIR
        self.collection_name = collection_name or Config.QDRANT_COLLECTION
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Set global embedding model
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # Initialize Qdrant client
        print(f"Connecting to Qdrant at {Config.QDRANT_HOST}:{Config.QDRANT_PORT}...")
        self.client = QdrantClient(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT
        )
        
    def create_sample_manuals(self):
        """Create sample maintenance manuals for testing."""
        self.manuals_dir.mkdir(parents=True, exist_ok=True)
        
        manuals = {
            "turbine_maintenance_guide.txt": """
TURBOFAN ENGINE MAINTENANCE GUIDE

1. VIBRATION MONITORING
- Normal vibration levels: < 0.5 in/sec
- High vibration (>0.7 in/sec) indicates:
  * Unbalanced rotor
  * Bearing degradation
  * Fan blade damage
- Corrective actions:
  * Inspect fan blades for FOD (Foreign Object Damage)
  * Check bearing clearances
  * Perform rotor balancing if needed

2. TEMPERATURE MONITORING
- Normal operating temperature (T30): 1580°F ± 20°F
- High temperature (>1620°F) causes:
  * Accelerated component wear
  * Reduced component life
  * Potential thermal distortion
- Required actions for high T30:
  * Reduce power setting
  * Inspect combustor for damage
  * Check fuel nozzle spray pattern
  * Verify bleed valve operation

3. PRESSURE RATIO (EPR) DEGRADATION
- Normal EPR: 1.30 ± 0.05
- Declining EPR indicates:
  * Compressor fouling
  * Turbine blade erosion
  * Internal leakage
- Maintenance procedures:
  * Perform compressor water wash
  * Borescope inspection of turbine
  * Check clearances at turbine seals

4. BLEED AIR SYSTEM
- Bleed enthalpy normal range: 390 ± 10 BTU/lb
- High bleed values suggest:
  * Bleed valve malfunction
  * Excessive secondary air flow
- Corrective maintenance:
  * Inspect bleed valve actuator
  * Check valve seat for wear
  * Verify control system operation

5. FUEL SYSTEM
- Normal fuel-air ratio (farB): 0.030 ± 0.002
- Rich mixture (>0.033) causes:
  * Incomplete combustion
  * Carbon buildup
  * High EGT
- Lean mixture (<0.027) causes:
  * Combustion instability
  * Potential flameout
- Maintenance actions:
  * Clean fuel nozzles
  * Check fuel pump output
  * Verify fuel control unit calibration

6. PREVENTIVE MAINTENANCE SCHEDULE
- Daily: Visual inspection, vibration check
- Weekly: Oil analysis, filter inspection
- Monthly: Borescope inspection
- Quarterly: Bearing inspection
- Annual: Major overhaul assessment
            """,
            
            "troubleshooting_procedures.txt": """
ENGINE TROUBLESHOOTING PROCEDURES

SYMPTOM: High Vibration Levels

Step 1: Immediate Actions
- Reduce power to idle
- Monitor vibration trend
- If vibration >1.0 in/sec, shut down engine

Step 2: Visual Inspection
- Inspect fan blades for damage
- Check for loose components
- Look for oil leaks near bearings

Step 3: Diagnostic Tests
- Perform vibration spectrum analysis
- Check bearing temperatures
- Measure shaft runout

Step 4: Corrective Measures
- Replace damaged fan blades
- Rebalance rotor assembly
- Replace worn bearings
- Realign engine mounts

SYMPTOM: Increasing Exhaust Temperature

Step 1: Immediate Actions
- Reduce power by 10%
- Monitor temperature trend
- Check for engine fire indications

Step 2: System Checks
- Verify thermocouple accuracy
- Check fuel flow rate
- Inspect combustor section

Step 3: Root Cause Analysis
- Deteriorated turbine blades
- Combustor damage
- Fuel nozzle blockage
- Cooling air bypass

Step 4: Repair Actions
- Replace damaged turbine blades
- Repair/replace combustor cans
- Clean or replace fuel nozzles
- Seal air leaks in cooling passages

SYMPTOM: Low Engine Pressure Ratio

Step 1: Performance Analysis
- Compare current EPR to baseline
- Check compressor efficiency
- Measure turbine gas temperature

Step 2: Compressor Assessment
- Perform compressor wash
- Inspect for erosion damage
- Check variable vane rigging

Step 3: Turbine Inspection
- Borescope turbine blades
- Measure blade dimensions
- Check for thermal distortion

Step 4: Restoration
- Replace eroded compressor blades
- Restore turbine blade profiles
- Adjust variable geometry
- Seal internal leakage paths
            """,
            
            "safety_procedures.txt": """
SAFETY PROCEDURES AND PRECAUTIONS

GENERAL SAFETY RULES
1. Always follow lockout/tagout procedures
2. Wear appropriate PPE (gloves, safety glasses, hearing protection)
3. Never work on running engines
4. Maintain clear communication during maintenance
5. Follow hot engine handling procedures

CRITICAL WARNINGS

⚠️ HIGH TEMPERATURE HAZARD
- Engine components remain hot for 30+ minutes after shutdown
- Exhaust gas temperatures can exceed 1400°F
- Allow adequate cooling time before touching components

⚠️ HIGH PRESSURE SYSTEMS
- Bleed air pressure: 400 psi
- Fuel system pressure: 3000 psi
- Always depressurize systems before disconnecting lines

⚠️ ROTATING MACHINERY
- Never approach running engine from front
- Maintain 25-foot safety zone around operating engine
- Secure all loose objects before engine start

⚠️ FUEL HANDLING
- No smoking or open flames within 50 feet
- Use proper grounding to prevent static discharge
- Clean up spills immediately
- Proper ventilation required

EMERGENCY PROCEDURES

Engine Fire:
1. Shut down engine immediately
2. Activate fire suppression system
3. Evacuate area
4. Call emergency services

Uncontained Failure:
1. Emergency engine shutdown
2. Clear immediate area
3. Inspect for secondary damage
4. Document debris field

Oil System Failure:
1. Reduce power immediately
2. Monitor bearing temperatures
3. Prepare for controlled shutdown
4. Do not restart without inspection
            """
        }
        
        for filename, content in manuals.items():
            file_path = self.manuals_dir / filename
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"  ✓ Created {filename}")
    
    def load_documents(self) -> List[Document]:
        """
        Load documents from manuals directory.
        
        Returns:
            List of Document objects
        """
        if not self.manuals_dir.exists() or not any(self.manuals_dir.iterdir()):
            print("No manuals found, creating samples...")
            self.create_sample_manuals()
        
        print(f"Loading documents from {self.manuals_dir}...")
        
        reader = SimpleDirectoryReader(
            input_dir=str(self.manuals_dir),
            recursive=True
        )
        
        documents = reader.load_data()
        print(f"✓ Loaded {len(documents)} documents")
        
        return documents
    
    def create_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name in collection_names:
            print(f"Collection '{self.collection_name}' already exists")
            return
        
        print(f"Creating collection '{self.collection_name}'...")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=Config.EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        
        print("✓ Collection created")
    
    def build_index(self, documents: List[Document] = None) -> VectorStoreIndex:
        """
        Build vector index from documents.
        
        Args:
            documents: Documents to index (loads if None)
            
        Returns:
            VectorStoreIndex
        """
        if documents is None:
            documents = self.load_documents()
        
        # Create collection
        self.create_collection()
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index
        print("Building vector index...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print("✓ Index built successfully")
        return index
    
    def verify_index(self):
        """Verify the index was created successfully."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            print(f"\n✓ Index verification:")
            print(f"  Collection: {self.collection_name}")
            print(f"  Vectors: {collection_info.vectors_count}")
            print(f"  Points: {collection_info.points_count}")
            return True
        except Exception as e:
            print(f"✗ Index verification failed: {e}")
            return False


def main():
    """Main index building pipeline."""
    print("=" * 60)
    print("RAG Index Builder")
    print("=" * 60)
    
    try:
        # Initialize builder
        builder = IndexBuilder()
        
        # Build index
        index = builder.build_index()
        
        # Verify
        builder.verify_index()
        
        print("\n" + "=" * 60)
        print("Index Building Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker run -p 6333:6333 qdrant/qdrant")


if __name__ == "__main__":
    main()
