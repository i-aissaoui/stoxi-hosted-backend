from typing import Optional
import hashlib

# Helpers
def _normalize_url_path(path: Optional[str]) -> Optional[str]:
    """Ensure URLs use forward slashes and have a single leading slash."""
    if not path:
        return path
    p = path.replace("\\", "/")
    if not p.startswith("/"):
        p = f"/{p}"
    # Collapse any accidental double slashes except the leading one
    while '//' in p[1:]:
        p = p[0] + p[1:].replace('//', '/')
    return p

def _sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None
"""
Hosted Backend v2 - Simplified (No Local Backend Calls)
Local backend will connect to us instead!
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
import shutil
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Algeria timezone
os.environ['TZ'] = 'Africa/Algiers'
ALGERIA_TZ = pytz.timezone('Africa/Algiers')

def get_algeria_time():
    """Get current time in Algeria timezone"""
    return datetime.now(ALGERIA_TZ)

def validate_image_file(filename: str) -> bool:
    """Validate if file is a supported image format"""
    if not filename:
        return False
    
    supported_extensions = (
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', 
        '.webp', '.tiff', '.tif', '.svg', '.ico', 
        '.heic', '.heif', '.avif', '.jfif', '.pjpeg', '.pjp'
    )
    
    return filename.lower().endswith(supported_extensions)

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

# Create static directories for images
os.makedirs("static/images/products", exist_ok=True)
os.makedirs("static/images/categories", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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
        # Normalize product image URL before serializing
        normalized_image = _normalize_url_path(product.image_url)
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
                "image_url": normalized_image,
                "variants": variants,
                "min_price": min(v["price"] for v in variants),
                "max_price": max(v["price"] for v in variants)
            })
    
    return {
        "success": True,
        "data": result,
        "count": len(result)
    }

@app.get("/admin/images/manifest")
async def images_manifest(current_user: User = Depends(get_current_user)):
    """Return a manifest of hosted static images (categories & products) with checksums.
    Used by local backend to verify and reconcile images.
    """
    base_dirs = [
        ("categories", "static/images/categories"),
        ("products", "static/images/products"),
    ]
    files = []
    try:
        for group, base in base_dirs:
            if not os.path.exists(base):
                continue
            for root, _, filenames in os.walk(base):
                for name in filenames:
                    full_path = os.path.join(root, name)
                    rel_path = os.path.relpath(full_path).replace("\\", "/")
                    checksum = _sha256_file(full_path)
                    stat = os.stat(full_path)
                    files.append({
                        "group": group,
                        "path": rel_path,
                        "url": _normalize_url_path(rel_path),
                        "sha256": checksum,
                        "size": stat.st_size,
                        "mtime": int(stat.st_mtime),
                    })
        return {"success": True, "files": files, "count": len(files)}
    except Exception as e:
        logger.error(f"❌ Error building images manifest: {e}")
        raise HTTPException(status_code=500, detail="Failed to build images manifest")

@app.post("/admin/images/delete")
async def delete_image_file(payload: Dict[str, Any], current_user: User = Depends(get_current_user)):
    """Delete a static image file by relative path under static/.
    Payload: {"path": "static/images/..."}
    """
    try:
        path = payload.get("path")
        if not path or not isinstance(path, str):
            raise HTTPException(status_code=400, detail="Invalid path")
        # Normalize and restrict to static directory
        norm = path.replace("\\", "/").lstrip("/")
        if not norm.startswith("static/"):
            raise HTTPException(status_code=400, detail="Path must be under static/")
        if ".." in norm:
            raise HTTPException(status_code=400, detail="Invalid path traversal")
        if not os.path.exists(norm):
            return {"success": True, "deleted": False, "message": "File not found"}
        os.remove(norm)
        logger.info(f"🗑️ Deleted static file: {norm}")
        return {"success": True, "deleted": True, "path": f"/{norm}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting static file: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file")

@app.get("/admin/export/users")
async def export_users(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Admin export of all users (for reconciliation)."""
    users = db.query(User).all()
    return {
        "success": True,
        "data": [
            {
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "is_active": u.is_active,
                "is_superuser": u.is_superuser,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "updated_at": u.updated_at.isoformat() if u.updated_at else None,
            }
            for u in users
        ],
        "count": len(users),
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

@app.post("/sync/users")
async def receive_user_update(user_data: Dict[str, Any], db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Receive user update from local backend"""
    global last_local_backend_ping
    last_local_backend_ping = get_algeria_time()
    
    try:
        user_id = user_data.get("id")
        operation = user_data.get("operation", "update")
        
        if operation == "delete":
            user = db.query(User).filter(User.id == user_id).first()
            if user and user.username != "owner":  # Don't delete owner user
                db.delete(user)
        else:
            existing = db.query(User).filter(User.id == user_id).first()
            
            if existing:
                # Update existing user (except owner username)
                for key, value in user_data.items():
                    if key not in ["id", "operation"] and not (key == "username" and existing.username == "owner"):
                        setattr(existing, key, value)
                existing.updated_at = get_algeria_time()
            else:
                # Create new user
                user_data.pop("operation", None)
                user_data["created_at"] = get_algeria_time()
                user_data["updated_at"] = get_algeria_time()
                user = User(**user_data)
                db.add(user)
        
        db.commit()
        logger.info(f"👤 Received user update: {user_id}")
        
        return {"success": True, "message": "User updated", "timestamp": get_algeria_time().isoformat()}
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error receiving user update: {e}")
        return {"success": False, "error": str(e)}

@app.post("/sync/stock")
async def receive_stock_update(stock_data: Dict[str, Any], db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Receive stock update from local backend"""
    global last_local_backend_ping
    last_local_backend_ping = get_algeria_time()
    
    try:
        variant_id = stock_data.get("variant_id")
        product_id = stock_data.get("product_id")
        
        # Find variant by ID or product_id
        variant = None
        if variant_id:
            variant = db.query(Variant).filter(Variant.id == variant_id).first()
        elif product_id:
            # Find first variant of the product
            variant = db.query(Variant).filter(Variant.product_id == product_id).first()
        
        if variant:
            # Update stock data
            variant.quantity = stock_data.get("quantity", variant.quantity)
            variant.price = stock_data.get("price", variant.price)
            variant.stopped = stock_data.get("stopped", variant.stopped)
            variant.last_synced = get_algeria_time()
            
            db.commit()
            logger.info(f"📦 Received stock update: variant {variant_id}, qty: {variant.quantity}")
        else:
            logger.warning(f"⚠️ Variant not found for stock update: {variant_id}")
        
        return {"success": True, "message": "Stock updated", "timestamp": get_algeria_time().isoformat()}
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error receiving stock update: {e}")
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
    """After local backend confirms receipt, delete these orders from hosted.
    This avoids duplication and keeps hosted queue clean.
    """
    try:
        orders = db.query(Order).filter(Order.id.in_(order_ids)).all()
        count = len(orders)
        for order in orders:
            db.delete(order)
        db.commit()
        logger.info(f"🗑️ Deleted {count} orders after local sync confirmation")
        return {
            "success": True,
            "message": f"Deleted {count} orders after sync confirmation",
            "deleted": count,
            "timestamp": get_algeria_time().isoformat(),
        }
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error deleting orders after sync confirmation: {e}")
        return {"success": False, "error": str(e)}

@app.get("/sync/queued-orders")
async def get_queued_orders_alt(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Alternative endpoint for queued orders (local backend compatibility)"""
    return await get_queued_orders(db, current_user)

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

# Image Transfer from Local Backend

@app.post("/sync/upload-image")
async def receive_image_from_local(
    file: UploadFile = File(...),
    image_path: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Receive image file from local backend during sync"""
    try:
        # Validate image file
        if not validate_image_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid image format. Supported: PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF, SVG, ICO, HEIC, HEIF, AVIF"
            )
        
        # Normalize the image path
        if image_path.startswith("/"):
            image_path = image_path[1:]
        
        # Create directory structure
        file_dir = os.path.dirname(image_path)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)
        
        # Save the file
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"📸 Received image from local backend: {image_path}")
        
        return {
            "success": True,
            "message": "Image received successfully",
            "path": f"/{image_path}"
        }
        
    except Exception as e:
        logger.error(f"❌ Error receiving image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to receive image: {str(e)}")

# Image Upload Endpoints

@app.post("/upload/category-image/{category_id}")
async def upload_category_image(
    category_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload image for category"""
    try:
        # Validate image file
        if not validate_image_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid image format. Supported: PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF, SVG, ICO, HEIC, HEIF, AVIF"
            )
        
        # Check if category exists
        category = db.query(Category).filter(Category.id == category_id).first()
        if not category:
            raise HTTPException(status_code=404, detail="Category not found")
        
        # Create directory
        upload_dir = "static/images/categories"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        ext = os.path.splitext(file.filename)[-1]
        filename = f"category_{category_id}{ext}"
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update category image URL
        category.image_url = f"/static/images/categories/{filename}"
        db.commit()
        
        logger.info(f"📸 Category {category_id} image uploaded: {filename}")
        
        return {
            "success": True,
            "image_url": category.image_url,
            "message": "Category image uploaded successfully"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error uploading category image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

@app.post("/upload/product-image/{product_id}")
async def upload_product_image(
    product_id: int,
    file: UploadFile = File(...),
    set_as_main: bool = Form(False),
    db: Session = Depends(get_db)
):
    """Upload image for product"""
    try:
        # Validate image file
        if not validate_image_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid image format. Supported: PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF, SVG, ICO, HEIC, HEIF, AVIF"
            )
        
        # Check if product exists
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Create directory for this product's images
        product_dir = f"static/images/products/product_{product_id}"
        os.makedirs(product_dir, exist_ok=True)
        
        # Check how many images already exist
        existing_files = os.listdir(product_dir) if os.path.exists(product_dir) else []
        image_number = len(existing_files) + 1
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"product_{image_number}{file_extension}"
        file_path = os.path.join(product_dir, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create image URL
        image_url = _normalize_url_path(file_path)
        
        # Update product image URL if it's the first image or set_as_main is True
        if set_as_main or image_number == 1:
            product.image_url = image_url
            db.commit()
        
        logger.info(f"📸 Product {product_id} image uploaded: {unique_filename}")
        
        return {
            "success": True,
            "image_url": image_url,
            "is_main": set_as_main or image_number == 1,
            "message": "Product image uploaded successfully"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error uploading product image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

@app.get("/images/product/{product_id}")
async def get_product_images(product_id: int, db: Session = Depends(get_db)):
    """Get all images for a product"""
    try:
        # Check if product exists
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Check product's image directory
        product_dir = f"static/images/products/product_{product_id}"
        
        if not os.path.exists(product_dir):
            return {"success": True, "images": [], "count": 0}
        
        # Get all image files (support all common formats)
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg', '.ico', '.heic', '.heif', '.avif')
        image_files = [f for f in os.listdir(product_dir) 
                      if os.path.isfile(os.path.join(product_dir, f)) and 
                      f.lower().endswith(image_extensions)]
        
        # Create response
        images = []
        for img in image_files:
            image_url = _normalize_url_path(os.path.join(product_dir, img))
            is_main = (product.image_url == image_url)
            images.append({
                "image_url": image_url,
                "is_main": is_main,
                "filename": img
            })
        
        return {
            "success": True,
            "images": images,
            "count": len(images)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting product images: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get images: {str(e)}")

@app.delete("/images/product/{product_id}")
async def delete_product_image(
    product_id: int,
    image_url: str,
    db: Session = Depends(get_db)
):
    """Delete a product image"""
    try:
        # Check if product exists
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Normalize image URL
        if not image_url.startswith("/"):
            image_url = f"/{image_url}"
        
        file_path = image_url[1:]  # Remove leading slash
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Check if it's the main image
        if product.image_url == image_url:
            # Find other images to set as main
            product_dir = f"static/images/products/product_{product_id}"
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg', '.ico', '.heic', '.heif', '.avif')
            other_images = [f for f in os.listdir(product_dir) 
                            if os.path.isfile(os.path.join(product_dir, f)) 
                            and f.lower().endswith(image_extensions)
                            and f"/{os.path.join(product_dir, f)}" != image_url]
            
            if other_images:
                # Set first alternative as main
                product.image_url = f"/{os.path.join(product_dir, other_images[0])}"
            else:
                # No other images, clear the image URL
                product.image_url = None
            
            db.commit()
        
        # Delete the file
        os.remove(file_path)
        
        logger.info(f"🗑️ Deleted product {product_id} image: {image_url}")
        
        return {
            "success": True,
            "message": "Image deleted successfully"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error deleting product image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)