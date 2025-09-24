"""
Hosted Backend v2 - Simplified (No Local Backend Calls)
Local backend will connect to us instead!
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Numeric, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime, timedelta
import os
import logging
from typing import List, Optional, Dict, Any
import json
import hashlib
import secrets
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Algeria timezone
os.environ['TZ'] = 'Africa/Algiers'
ALGERIA_TZ = pytz.timezone('Africa/Algiers')

def get_algeria_time():
    """Get current time in Algeria timezone"""
    return datetime.now(ALGERIA_TZ)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hosted_shiakati.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text, nullable=False)
    description = Column(Text)
    image_url = Column(Text)
    created_at = Column(DateTime, default=get_algeria_time)
    last_synced = Column(DateTime, default=get_algeria_time)
    
    products = relationship("Product", back_populates="category")

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text, nullable=False)
    description = Column(Text)
    category_id = Column(Integer, ForeignKey("categories.id"))
    image_url = Column(Text)
    show_on_website = Column(Integer, default=1)
    created_at = Column(DateTime, default=get_algeria_time)
    last_synced = Column(DateTime, default=get_algeria_time)
    
    category = relationship("Category", back_populates="products")
    variants = relationship("Variant", back_populates="product")

class Variant(Base):
    __tablename__ = "variants"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    barcode = Column(Text)
    price = Column(Numeric(10, 2), nullable=False)
    quantity = Column(Numeric(10, 3), default=0)
    width = Column(Text)
    height = Column(Text)
    color = Column(Text)
    stopped = Column(Boolean, default=False)
    created_at = Column(DateTime, default=get_algeria_time)
    last_synced = Column(DateTime, default=get_algeria_time)
    
    product = relationship("Product", back_populates="variants")

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text, nullable=False)
    phone_number = Column(Text)
    created_at = Column(DateTime, default=get_algeria_time)

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    customer_name = Column(Text)
    customer_phone = Column(Text)
    total = Column(Numeric(10, 2), nullable=False)
    status = Column(Text, default="pending")
    delivery_method = Column(Text)
    wilaya = Column(Text)
    commune = Column(Text)
    address = Column(Text)
    notes = Column(Text)
    order_time = Column(DateTime, default=get_algeria_time)
    synced_to_local = Column(Boolean, default=False)
    local_order_id = Column(Integer)
    
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

class OrderItem(Base):
    __tablename__ = "order_items"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    variant_id = Column(Integer)
    product_name = Column(Text)
    variant_details = Column(Text)
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(10, 2), nullable=False)
    
    order = relationship("Order", back_populates="items")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=get_algeria_time)
    updated_at = Column(DateTime, default=get_algeria_time, onupdate=get_algeria_time)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI app
app = FastAPI(
    title="Shiakati Store - Hosted Backend v2",
    description="Website backend - Local backend connects to us",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
last_local_backend_ping = None

# Authentication setup
security = HTTPBearer()
API_KEY = os.getenv("API_KEY", "123456789")  # Change in production

def authenticate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Authenticate using API key"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

def get_current_user(db: Session = Depends(get_db), credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    authenticate_api_key(credentials)
    return User(id=0, username="api_user", email="api@system.com", is_active=True)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Starting Hosted Backend v2 (Algeria GMT+1)")

# API Endpoints

@app.get("/")
async def root(db: Session = Depends(get_db)):
    """Root endpoint with status"""
    return {
        "message": "Shiakati Store - Hosted Backend v2",
        "status": "running",
        "timezone": "Africa/Algiers (GMT+1)",
        "current_time": get_algeria_time().isoformat(),
        "last_local_ping": last_local_backend_ping.isoformat() if last_local_backend_ping else None,
        "categories": db.query(Category).count(),
        "products": db.query(Product).count(),
        "pending_orders": db.query(Order).filter(Order.synced_to_local == False).count()
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy", 
        "timestamp": get_algeria_time().isoformat(),
        "timezone": "Africa/Algiers"
    }

@app.get("/categories")
async def get_categories(db: Session = Depends(get_db)):
    """Get categories for website"""
    categories = db.query(Category).all()
    
    return {
        "success": True,
        "data": [
            {
                "id": cat.id,
                "name": cat.name,
                "description": cat.description,
                "image_url": cat.image_url,
                "created_at": cat.created_at.isoformat() if cat.created_at else None
            }
            for cat in categories
        ],
        "count": len(categories)
    }

@app.get("/products")
async def get_products(
    category_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50),
    db: Session = Depends(get_db)
):
    """Get products for website"""
    query = db.query(Product).filter(Product.show_on_website == 1)
    
    if category_id:
        query = query.filter(Product.category_id == category_id)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(Product.name.ilike(search_term))
    
    products = query.limit(limit).all()
    
    result = []
    for product in products:
        variants = [
            {
                "id": v.id,
                "price": float(v.price),
                "quantity": float(v.quantity),
                "width": v.width,
                "height": v.height,
                "color": v.color,
                "available": float(v.quantity) > 0 and not v.stopped
            }
            for v in product.variants
            if not v.stopped
        ]
        
        if variants:
            result.append({
                "id": product.id,
                "name": product.name,
                "description": product.description,
                "category_id": product.category_id,
                "category_name": product.category.name if product.category else None,
                "image_url": product.image_url,
                "variants": variants,
                "min_price": min(v["price"] for v in variants),
                "max_price": max(v["price"] for v in variants)
            })
    
    return {
        "success": True,
        "data": result,
        "count": len(result)
    }

@app.post("/orders")
async def create_order(order_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Create order from website - queued for local backend"""
    global last_local_backend_ping
    
    try:
        # Create customer if needed
        customer = None
        if order_data.get("customer_phone"):
            customer = db.query(Customer).filter(
                Customer.phone_number == order_data["customer_phone"]
            ).first()
            
            if not customer:
                customer = Customer(
                    name=order_data.get("customer_name", ""),
                    phone_number=order_data["customer_phone"]
                )
                db.add(customer)
                db.flush()
        
        # Create order with Algeria time
        order = Order(
            customer_id=customer.id if customer else None,
            customer_name=order_data.get("customer_name", ""),
            customer_phone=order_data.get("customer_phone", ""),
            total=order_data.get("total", 0),
            delivery_method=order_data.get("delivery_method", "home_delivery"),
            wilaya=order_data.get("wilaya", ""),
            commune=order_data.get("commune", ""),
            address=order_data.get("address", ""),
            notes=order_data.get("notes", "Order from website"),
            order_time=get_algeria_time(),
            synced_to_local=False
        )
        
        db.add(order)
        db.flush()
        
        # Add order items
        for item_data in order_data.get("items", []):
            order_item = OrderItem(
                order_id=order.id,
                variant_id=item_data["variant_id"],
                product_name=item_data.get("product_name", ""),
                variant_details=json.dumps(item_data.get("variant_details", {})),
                quantity=item_data["quantity"],
                price=item_data["price"]
            )
            db.add(order_item)
        
        db.commit()
        
        logger.info(f"📝 Created website order {order.id} at {get_algeria_time()}")
        
        return {
            "success": True,
            "message": "Order created successfully",
            "order_id": order.id,
            "order_time": get_algeria_time().isoformat(),
            "queued_for_local": True
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Order creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create order: {str(e)}")

# Sync endpoints for local backend to call

@app.post("/sync/ping")
async def local_backend_ping(current_user: User = Depends(get_current_user)):
    """Local backend pings to show it's online"""
    global last_local_backend_ping
    last_local_backend_ping = get_algeria_time()
    logger.info("📡 Local backend pinged - connection active")
    return {"success": True, "message": "Ping received", "server_time": get_algeria_time().isoformat()}

@app.post("/sync/products")
async def receive_product_update(product_data: Dict[str, Any], db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Receive product update from local backend"""
    global last_local_backend_ping
    last_local_backend_ping = get_algeria_time()
    
    try:
        product_id = product_data.get("id")
        operation = product_data.get("operation", "update")
        
        if operation == "delete":
            product = db.query(Product).filter(Product.id == product_id).first()
            if product:
                db.delete(product)
        else:
            existing = db.query(Product).filter(Product.id == product_id).first()
            
            if existing:
                # Update existing product
                for key, value in product_data.items():
                    if key not in ["id", "variants", "operation"]:
                        setattr(existing, key, value)
                existing.last_synced = get_algeria_time()
                
                # Update variants
                for variant_data in product_data.get("variants", []):
                    existing_variant = db.query(Variant).filter(Variant.id == variant_data["id"]).first()
                    if existing_variant:
                        for key, value in variant_data.items():
                            if key != "id":
                                setattr(existing_variant, key, value)
                        existing_variant.last_synced = get_algeria_time()
                    else:
                        variant = Variant(**variant_data, last_synced=get_algeria_time())
                        db.add(variant)
            else:
                # Create new product
                variants_data = product_data.pop("variants", [])
                product_data.pop("operation", None)
                product = Product(**product_data, last_synced=get_algeria_time())
                db.add(product)
                db.flush()
                
                # Create variants
                for variant_data in variants_data:
                    variant_data["product_id"] = product.id
                    variant = Variant(**variant_data, last_synced=get_algeria_time())
                    db.add(variant)
        
        db.commit()
        logger.info(f"📥 Received product update: {product_id}")
        
        return {"success": True, "message": "Product updated", "timestamp": get_algeria_time().isoformat()}
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error receiving product update: {e}")
        return {"success": False, "error": str(e)}

@app.post("/sync/categories")
async def receive_category_update(category_data: Dict[str, Any], db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Receive category update from local backend"""
    global last_local_backend_ping
    last_local_backend_ping = get_algeria_time()
    
    try:
        category_id = category_data.get("id")
        operation = category_data.get("operation", "update")
        
        if operation == "delete":
            category = db.query(Category).filter(Category.id == category_id).first()
            if category:
                db.delete(category)
        else:
            existing = db.query(Category).filter(Category.id == category_id).first()
            
            if existing:
                for key, value in category_data.items():
                    if key not in ["id", "operation"]:
                        setattr(existing, key, value)
                existing.last_synced = get_algeria_time()
            else:
                category_data.pop("operation", None)
                category = Category(**category_data, last_synced=get_algeria_time())
                db.add(category)
        
        db.commit()
        logger.info(f"📁 Received category update: {category_id}")
        
        return {"success": True, "message": "Category updated", "timestamp": get_algeria_time().isoformat()}
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error receiving category update: {e}")
        return {"success": False, "error": str(e)}

@app.get("/sync/orders")
async def get_queued_orders(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get orders queued for local backend"""
    global last_local_backend_ping
    last_local_backend_ping = get_algeria_time()
    
    try:
        orders = db.query(Order).filter(Order.synced_to_local == False).all()
        
        orders_data = []
        for order in orders:
            orders_data.append({
                "id": order.id,
                "customer_name": order.customer_name,
                "customer_phone": order.customer_phone,
                "wilaya": order.wilaya,
                "commune": order.commune,
                "address": order.address,
                "delivery_method": order.delivery_method,
                "notes": order.notes,
                "total": float(order.total),
                "order_time": order.order_time.isoformat() if order.order_time else None,
                "items": [
                    {
                        "variant_id": item.variant_id,
                        "quantity": item.quantity,
                        "price": float(item.price),
                        "product_name": item.product_name,
                        "variant_details": json.loads(item.variant_details) if item.variant_details else {}
                    }
                    for item in order.items
                ]
            })
        
        logger.info(f"📤 Sent {len(orders_data)} queued orders to local backend")
        return {"success": True, "orders": orders_data, "timestamp": get_algeria_time().isoformat()}
        
    except Exception as e:
        logger.error(f"❌ Error getting queued orders: {e}")
        return {"success": False, "error": str(e)}

@app.post("/sync/mark-orders-synced")
async def mark_orders_synced(order_ids: List[int], db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Mark orders as synced to local backend"""
    try:
        orders = db.query(Order).filter(Order.id.in_(order_ids)).all()
        
        for order in orders:
            order.synced_to_local = True
        
        db.commit()
        logger.info(f"✅ Marked {len(orders)} orders as synced")
        
        return {"success": True, "message": f"Marked {len(orders)} orders as synced", "timestamp": get_algeria_time().isoformat()}
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error marking orders as synced: {e}")
        return {"success": False, "error": str(e)}

@app.get("/sync/status")
async def sync_status(db: Session = Depends(get_db)):
    """Get sync status"""
    pending_orders = db.query(Order).filter(Order.synced_to_local == False).count()
    
    local_online = False
    if last_local_backend_ping:
        time_since_ping = (get_algeria_time() - last_local_backend_ping).total_seconds()
        local_online = time_since_ping < 120  # Consider online if pinged within 2 minutes
    
    return {
        "local_backend_online": local_online,
        "last_local_ping": last_local_backend_ping.isoformat() if last_local_backend_ping else None,
        "pending_orders": pending_orders,
        "current_time": get_algeria_time().isoformat(),
        "timezone": "Africa/Algiers",
        "mode": "passive_receiver"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)