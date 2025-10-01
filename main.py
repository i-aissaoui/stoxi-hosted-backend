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
    # New: delivery fee stored to allow correct totals and syncing
    delivery_fee = Column(Numeric(10, 2))
    
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

class DeliveryFee(Base):
    __tablename__ = "delivery_fees"
    
    id = Column(Integer, primary_key=True, index=True)
    wilaya_code = Column(Integer, nullable=False, index=True)
    wilaya_name = Column(Text, nullable=False)
    home_delivery_fee = Column(Numeric(10, 2), nullable=False)
    office_delivery_fee = Column(Numeric(10, 2))

# Create tables
Base.metadata.create_all(bind=engine)

# Ensure new columns exist on existing deployments (best-effort)
try:
    with engine.connect() as conn:
        # Postgres supports IF NOT EXISTS; SQLite will ignore a failing clause, so we try-catch
        conn.execute("ALTER TABLE orders ADD COLUMN IF NOT EXISTS delivery_fee NUMERIC(10,2)")
        conn.commit()
except Exception:
    try:
        # Fallback: try simple ADD COLUMN (may fail if already exists)
        with engine.connect() as conn:
            conn.execute("ALTER TABLE orders ADD COLUMN delivery_fee NUMERIC(10,2)")
            conn.commit()
    except Exception:
        pass

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
    """Get ONLY categories that have at least one visible product (show_on_website=1)."""
    # Join with products and keep distinct categories that have visible products
    categories = (
        db.query(Category)
        .join(Product, Product.category_id == Category.id)
        .filter(Product.show_on_website == 1)
        .distinct()
        .all()
    )
    
    data = [
        {
            "id": cat.id,
            "name": cat.name,
            "description": cat.description,
            "image_url": cat.image_url,
            "created_at": cat.created_at.isoformat() if cat.created_at else None,
        }
        for cat in categories
    ]
    return {"success": True, "data": data, "count": len(data)}

@app.get("/delivery-fees")
async def get_delivery_fees(db: Session = Depends(get_db)):
    """Public endpoint for the website to fetch delivery fees."""
    fees = db.query(DeliveryFee).all()
    items = [
        {
            "id": f.id,
            "wilaya_code": f.wilaya_code,
            "wilaya_name": f.wilaya_name,
            "home_delivery_fee": float(f.home_delivery_fee) if f.home_delivery_fee is not None else None,
            "office_delivery_fee": float(f.office_delivery_fee) if f.office_delivery_fee is not None else None,
        }
        for f in fees
    ]
    return {"success": True, "items": items, "count": len(items)}

@app.post("/sync/delivery-fees")
async def receive_delivery_fees_update(payload: Dict[str, Any] | List[Dict[str, Any]], db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Upsert delivery fee rows from the local backend.
    Accepts either a single object or a list of objects with fields:
    { id?, wilaya_code, wilaya_name, home_delivery_fee, office_delivery_fee }
    """
    try:
        # Normalize payload to a list
        items: List[Dict[str, Any]]
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = [payload]
        else:
            raise HTTPException(status_code=400, detail="Invalid payload type (expected list or object)")

        upserted = 0
        for it in items:
            code = it.get("wilaya_code")
            name = it.get("wilaya_name")
            if code is None or name is None:
                continue
            fee = db.query(DeliveryFee).filter(DeliveryFee.wilaya_code == code).first()
            if fee:
                fee.wilaya_name = name
                if it.get("home_delivery_fee") is not None:
                    try:
                        fee.home_delivery_fee = float(it["home_delivery_fee"])  # type: ignore
                    except Exception:
                        pass
                if it.get("office_delivery_fee") is not None:
                    try:
                        fee.office_delivery_fee = float(it["office_delivery_fee"])  # type: ignore
                    except Exception:
                        pass
            else:
                try:
                    hdf = it.get("home_delivery_fee", 0)
                    odf = it.get("office_delivery_fee")
                    fee = DeliveryFee(
                        wilaya_code=int(code),
                        wilaya_name=str(name),
                        home_delivery_fee=float(hdf) if hdf is not None else 0,
                        office_delivery_fee=float(odf) if odf is not None else None,
                    )
                    db.add(fee)
                except Exception:
                    continue
            upserted += 1
        db.commit()
        return {"success": True, "upserted": upserted}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error receiving delivery fees update: {e}")
        raise HTTPException(status_code=500, detail="Failed to update delivery fees")

@app.get("/products")
async def get_products(
    category_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(40, ge=1, le=200),
    skip: Optional[int] = Query(None, ge=0),
    include_stopped: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Paginated products list for the website.
    - Filters: category_id, search
    - Pagination: page (1-based), limit (page size), skip (optional overrides page)
    - Only returns products with show_on_website=1.
    """
    base_query = db.query(Product).filter(Product.show_on_website == 1)

    if category_id is not None:
        base_query = base_query.filter(Product.category_id == category_id)

    if search:
        search_term = f"%{search}%"
        base_query = base_query.filter(Product.name.ilike(search_term))

    # Total BEFORE pagination (for metadata)
    total = base_query.count()

    # Determine offset
    offset = skip if skip is not None else (page - 1) * limit

    # Order by newest first for better UX
    page_q = base_query.order_by(Product.created_at.desc(), Product.id.desc()).offset(offset).limit(limit)
    products = page_q.all()

    data = []
    for product in products:
        normalized_image = _normalize_url_path(product.image_url)
        # Build variants; optionally include stopped variants for completeness
        v_source = product.variants
        payload_variants = []
        for v in v_source:
            if not include_stopped and v.stopped:
                continue
            payload_variants.append({
                "id": v.id,
                "price": float(v.price),
                "quantity": float(v.quantity),
                "width": v.width,
                "height": v.height,
                "color": v.color,
                "stopped": bool(v.stopped),
                "available": float(v.quantity) > 0 and not v.stopped,
            })
        # Keep products that have at least one variant to display
        if payload_variants:
            prices = [pv["price"] for pv in payload_variants if pv.get("price") is not None]
            data.append({
                "id": product.id,
                "name": product.name,
                "description": product.description,
                "category_id": product.category_id,
                "category_name": product.category.name if product.category else None,
                "image_url": normalized_image,
                "variants": payload_variants,
                "min_price": min(prices) if prices else None,
                "max_price": max(prices) if prices else None,
            })

    pages = (total + limit - 1) // limit if limit else 1
    current_page = (offset // limit) + 1 if limit else 1
    return {
        "success": True,
        "data": data,
        "count": len(data),
        "total": total,
        "page": current_page,
        "page_size": limit,
        "pages": pages,
        "has_next": current_page < pages,
        "has_prev": current_page > 1,
    }

@app.get("/products/{product_id}")
async def get_product_detail(product_id: int, db: Session = Depends(get_db)):
    """Product detail with variants for the website."""
    product = db.query(Product).filter(Product.id == product_id, Product.show_on_website == 1).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    normalized_image = _normalize_url_path(product.image_url)
    variants = [
        {
            "id": v.id,
            "price": float(v.price),
            "quantity": float(v.quantity),
            "width": v.width,
            "height": v.height,
            "color": v.color,
            "stopped": bool(v.stopped),
            "available": float(v.quantity) > 0 and not v.stopped,
        }
        for v in product.variants
        if not v.stopped
    ]
    return {
        "success": True,
        "data": {
            "id": product.id,
            "name": product.name,
            "description": product.description,
            "category_id": product.category_id,
            "category_name": product.category.name if product.category else None,
            "image_url": normalized_image,
            "variants": variants,
        }
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

@app.post("/admin/images/set-main")
async def set_main_image(payload: Dict[str, Any], db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Set a product's main image (image_url on product)."""
    try:
        product_id = payload.get("product_id")
        image_url = payload.get("image_url")
        if not isinstance(product_id, int) or not isinstance(image_url, str):
            raise HTTPException(status_code=400, detail="Invalid payload")
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        product.image_url = _normalize_url_path(image_url)
        db.commit()
        return {"success": True, "image_url": product.image_url}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error setting main image: {e}")
        raise HTTPException(status_code=500, detail="Failed to set main image")

@app.get("/admin/export/categories")
async def export_categories(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Export all categories for reconciliation."""
    cats = db.query(Category).all()
    return {
        "success": True,
        "data": [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "image_url": c.image_url,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "last_synced": c.last_synced.isoformat() if getattr(c, "last_synced", None) else None,
            }
            for c in cats
        ],
        "count": len(cats),
    }

@app.get("/admin/export/products")
async def export_products(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Export all products with variants for reconciliation."""
    prods = db.query(Product).all()
    variants = db.query(Variant).all()
    var_by_pid: Dict[int, List[Variant]] = {}
    for v in variants:
        var_by_pid.setdefault(v.product_id, []).append(v)
    def v_to_dict(v: Variant) -> Dict[str, Any]:
        return {
            "id": v.id,
            "product_id": v.product_id,
            "width": v.width,
            "height": v.height,
            "color": v.color,
            "barcode": v.barcode,
            "price": float(v.price) if getattr(v, "price", None) is not None else None,
            # Hosted Variant may not have these optional fields; guard with getattr
            "cost_price": float(getattr(v, "cost_price")) if getattr(v, "cost_price", None) is not None else None,
            "quantity": float(getattr(v, "quantity", 0)) if getattr(v, "quantity", None) is not None else 0,
            "created_at": v.created_at.isoformat() if v.created_at else None,
            "promotion_price": float(getattr(v, "promotion_price")) if getattr(v, "promotion_price", None) is not None else None,
        }
    data = []
    for p in prods:
        data.append({
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "category_id": p.category_id,
            "image_url": p.image_url,
            "show_on_website": p.show_on_website,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "last_synced": p.last_synced.isoformat() if getattr(p, "last_synced", None) else None,
            "variants": [v_to_dict(v) for v in var_by_pid.get(p.id, [])],
        })
    return {"success": True, "data": data, "count": len(data)}

@app.delete("/sync/orders/{order_id}")
async def delete_synced_order(order_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Delete an order after local backend confirms receipt."""
    try:
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            return {"success": True, "deleted": False}
        db.delete(order)
        db.commit()
        return {"success": True, "deleted": True}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error deleting order {order_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete order")

@app.get("/sync/queued-orders")
async def get_queued_orders(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Return orders that are not yet synced to the local backend, with items.
    This is consumed by the local backend during manual/full sync.
    """
    try:
        orders = db.query(Order).filter(Order.synced_to_local == False).order_by(Order.id.asc()).limit(200).all()
        out = []
        for o in orders:
            out.append({
                "id": o.id,
                "customer_name": o.customer_name,
                "customer_phone": o.customer_phone,
                "total": float(o.total) if o.total is not None else 0,
                "delivery_fee": float(o.delivery_fee) if getattr(o, 'delivery_fee', None) is not None else None,
                "status": o.status,
                "delivery_method": o.delivery_method,
                "wilaya": o.wilaya,
                "commune": o.commune,
                "address": o.address,
                "notes": o.notes,
                "order_time": o.order_time.isoformat() if o.order_time else None,
                "items": [
                    (
                        (lambda _it: (
                            {
                                "variant_id": _it.variant_id,
                                "quantity": _it.quantity,
                                "price": float(_it.price) if _it.price is not None else 0,
                                # Forward variant details to help local resolve variants when IDs differ
                                "variant_details": (lambda _raw: (
                                    (lambda _parsed: _parsed if isinstance(_parsed, dict) else None)(
                                        (lambda: __import__("json").loads(_raw))() if isinstance(_raw, str) else (_raw if isinstance(_raw, dict) else None)
                                    )
                                ))(getattr(_it, 'variant_details', None)),
                                # Also include product name for matching
                                "product_name": getattr(_it, 'product_name', None),
                            }
                        ))(it)
                    )
                    for it in o.items
                ],
            })
        return {"success": True, "orders": out, "count": len(out)}
    except Exception as e:
        logger.error(f"❌ Error listing queued orders: {e}")
        raise HTTPException(status_code=500, detail="Failed to list queued orders")

@app.post("/sync/mark-orders-synced")
async def mark_orders_synced(order_ids: List[int], db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Mark provided hosted orders as synced to local (id list in body)."""
    try:
        if not isinstance(order_ids, list):
            raise HTTPException(status_code=400, detail="Body must be a list of order IDs")
        updated = 0
        for oid in order_ids:
            o = db.query(Order).filter(Order.id == int(oid)).first()
            if not o:
                continue
            o.synced_to_local = True
            updated += 1
        db.commit()
        return {"success": True, "updated": updated}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error marking orders synced: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark orders synced")

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
        # Normalize incoming fields (accept both customer_phone and phone_number)
        incoming_name = (order_data.get("customer_name") or "").strip()
        incoming_phone = (order_data.get("customer_phone") or order_data.get("phone_number") or "").strip()

        # Create or link customer by phone (unicity)
        customer = None
        if incoming_phone:
            customer = db.query(Customer).filter(
                Customer.phone_number == incoming_phone
            ).first()

            if not customer:
                customer = Customer(
                    name=incoming_name,
                    phone_number=incoming_phone
                )
                db.add(customer)
                db.flush()
            else:
                # If we have a better name now, update empty/generic names
                if incoming_name and (not customer.name or customer.name.strip() in {"", "Guest", "Website Guest", "-"}):
                    customer.name = incoming_name
                    db.flush()
        
        # Create order with Algeria time (total will be finalized after items loop)
        incoming_total = order_data.get("total", 0)
        try:
            incoming_total = float(incoming_total or 0)
        except Exception:
            incoming_total = 0.0
        # Capture delivery_fee from payload if provided
        try:
            incoming_delivery_fee = order_data.get("delivery_fee", None)
            incoming_delivery_fee = float(incoming_delivery_fee) if incoming_delivery_fee is not None else None
        except Exception:
            incoming_delivery_fee = None

        order = Order(
            customer_id=customer.id if customer else None,
            customer_name=incoming_name,
            customer_phone=incoming_phone,
            total=incoming_total,
            delivery_method=order_data.get("delivery_method", "home_delivery"),
            wilaya=order_data.get("wilaya", ""),
            commune=order_data.get("commune", ""),
            address=order_data.get("address", ""),
            notes=order_data.get("notes", "Order from website"),
            order_time=get_algeria_time(),
            synced_to_local=False,
            delivery_fee=incoming_delivery_fee
        )
        
        db.add(order)
        db.flush()
        
        # Add order items and compute total when prices are missing
        computed_total = 0.0
        for item_data in order_data.get("items", []):
            # Defensive parsing to avoid KeyError on missing fields
            try:
                variant_id = item_data.get("variant_id")
                qty = item_data.get("quantity", 1)
                # Accept alternative price keys and default to 0 if missing
                price_val = (
                    item_data.get("price", None)
                    if isinstance(item_data, dict) else None
                )
                if price_val is None:
                    price_val = item_data.get("unit_price") if isinstance(item_data, dict) else None
                if price_val is None:
                    price_val = item_data.get("variant_price") if isinstance(item_data, dict) else None
                if price_val is None:
                    # Try to fetch variant price from DB when not provided
                    try:
                        v = db.query(Variant).filter(Variant.id == int(variant_id)).first() if variant_id else None
                        price_val = float(v.price) if v and getattr(v, "price", None) is not None else 0
                    except Exception:
                        price_val = 0
                # Coerce types
                qty = int(qty) if qty is not None else 1
                try:
                    price_val = float(price_val)
                except Exception:
                    price_val = 0.0

                # Ensure we carry rich details to help local resolve variants
                details = item_data.get("variant_details", {}) or {}
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except Exception:
                        details = {}
                if not isinstance(details, dict):
                    details = {}

                # Enrich details and product_name from Variant if missing
                v_for_details = None
                try:
                    if variant_id:
                        v_for_details = db.query(Variant).filter(Variant.id == int(variant_id)).first()
                except Exception:
                    v_for_details = None
                if v_for_details is not None:
                    # Fill attributes if absent
                    details.setdefault("barcode", getattr(v_for_details, "barcode", None))
                    details.setdefault("width", getattr(v_for_details, "width", None))
                    details.setdefault("height", getattr(v_for_details, "height", None))
                    details.setdefault("color", getattr(v_for_details, "color", None))
                    # Also store product_name inside details for redundancy
                    if getattr(v_for_details, "product", None):
                        details.setdefault("product_name", getattr(v_for_details.product, "name", None))

                order_item = OrderItem(
                    order_id=order.id,
                    variant_id=variant_id,
                    product_name=(
                        item_data.get("product_name")
                        or (getattr(v_for_details.product, "name", None) if (v_for_details and getattr(v_for_details, "product", None)) else "")
                        or ""
                    ),
                    variant_details=json.dumps(details),
                    quantity=qty,
                    price=price_val,
                )
                db.add(order_item)
                # Accumulate total
                computed_total += (price_val or 0) * (qty or 0)
            except Exception as ie:
                logger.error(f"Order item parse error: {ie} | item={item_data}")
                # Skip invalid item but continue others
                continue

        # Commit items
        db.commit()

        # Finalize/normalize total: enforce sum(items) + delivery_fee
        try:
            normalized_total = float(computed_total or 0)
            if incoming_delivery_fee is not None:
                try:
                    normalized_total += float(incoming_delivery_fee)
                except Exception:
                    pass
            order.total = normalized_total
            db.commit()
        except Exception:
            pass

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
        category_id = product_data.get("category_id")

        # Normalize image_url if present (avoid backslashes)
        if isinstance(product_data.get("image_url"), str):
            product_data["image_url"] = _normalize_url_path(product_data["image_url"])  # type: ignore
        
        if operation == "delete":
            product = db.query(Product).filter(Product.id == product_id).first()
            if product:
                db.delete(product)
        else:
            existing = db.query(Product).filter(Product.id == product_id).first()
            
            if existing:
                # Update existing product
                # If changing category, validate it exists first
                if category_id is not None and category_id != existing.category_id:
                    if not db.query(Category).filter(Category.id == category_id).first():
                        msg = f"Cannot update product {product_id}: category_id {category_id} not found. Sync categories first."
                        logger.error(msg)
                        return {"success": False, "error": msg}, 400

                for key, value in product_data.items():
                    if key not in ["id", "variants", "operation"]:
                        setattr(existing, key, value)
                existing.last_synced = get_algeria_time()
                
                # Update variants
                # Helper: allow only attributes that exist on Variant model
                def _filter_variant_fields(d: dict) -> dict:
                    allowed = {}
                    for k, v in d.items():
                        if k == "id":
                            allowed[k] = v
                            continue
                        if hasattr(Variant, k):
                            allowed[k] = v
                    return allowed

                for variant_data in product_data.get("variants", []):
                    vdata = _filter_variant_fields(variant_data)
                    existing_variant = db.query(Variant).filter(Variant.id == vdata.get("id")).first()
                    if existing_variant:
                        for key, value in vdata.items():
                            if key != "id":
                                setattr(existing_variant, key, value)
                        existing_variant.last_synced = get_algeria_time()
                    else:
                        vdata.pop("id", None)  # id is PK; ensure explicit when creating if needed
                        variant = Variant(**vdata, last_synced=get_algeria_time())
                        db.add(variant)
                # Delete variants that are not present in payload
                payload_variant_ids = {v["id"] for v in product_data.get("variants", []) if "id" in v}
                to_delete = db.query(Variant).filter(Variant.product_id == existing.id).all()
                for ev in to_delete:
                    if ev.id not in payload_variant_ids:
                        db.delete(ev)
            else:
                # Create new product
                # Validate category exists before insert to avoid FK error
                if category_id is not None and not db.query(Category).filter(Category.id == category_id).first():
                    msg = f"Cannot create product {product_id}: category_id {category_id} not found. Sync categories first."
                    logger.error(msg)
                    return {"success": False, "error": msg}, 400

                variants_data = product_data.pop("variants", [])
                product_data.pop("operation", None)
                product = Product(**product_data, last_synced=get_algeria_time())
                db.add(product)
                db.flush()
                
                # Create variants
                # Reuse the same field filter used in update branch
                def _filter_variant_fields_create(d: dict) -> dict:
                    allowed = {}
                    for k, v in d.items():
                        if k == "id":
                            # ignore id when creating if provided
                            continue
                        if hasattr(Variant, k):
                            allowed[k] = v
                    return allowed

                for variant_data in variants_data:
                    vdata = _filter_variant_fields_create(variant_data)
                    vdata["product_id"] = product.id
                    variant = Variant(**vdata, last_synced=get_algeria_time())
                    db.add(variant)
        
        db.commit()
        logger.info(f"📥 Received product update: {product_id}")
        
        return {"success": True, "message": "Product updated", "timestamp": get_algeria_time().isoformat()}
        
    except Exception as e:
        db.rollback()
        try:
            # Provide more context in logs and response
            op = product_data.get("operation", "update")
            pid = product_data.get("id")
            cid = product_data.get("category_id")
            logger.error(
                "❌ Product sync failed | op=%s id=%s category_id=%s | %s: %s",
                op,
                pid,
                cid,
                e.__class__.__name__,
                str(e),
            )
            # Check if this looks like an FK error and hint the fix
            if "ForeignKeyViolation" in e.__class__.__name__ or "foreign key" in str(e).lower():
                return {
                    "success": False,
                    "error": "Category missing on hosted",
                    "details": f"category_id {cid} not found. Sync categories before products.",
                }, 400
        except Exception:
            pass
        return {"success": False, "error": str(e)}, 500

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

@app.post("/sync/delivery-fees")
async def receive_delivery_fees_update(payload: Any, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Upsert delivery fee rows from the local backend.
    Accepts either a single object or a list of objects with fields:
    { id?, wilaya_code, wilaya_name, home_delivery_fee, office_delivery_fee }
    """
    try:
        items: List[Dict[str, Any]]
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = [payload]
        else:
            raise HTTPException(status_code=400, detail="Invalid payload")

        upserted = 0
        for it in items:
            code = it.get("wilaya_code")
            name = it.get("wilaya_name")
            if code is None or name is None:
                continue
            fee = db.query(DeliveryFee).filter(DeliveryFee.wilaya_code == code).first()
            if fee:
                fee.wilaya_name = name
                if it.get("home_delivery_fee") is not None:
                    fee.home_delivery_fee = it["home_delivery_fee"]
                if it.get("office_delivery_fee") is not None:
                    fee.office_delivery_fee = it["office_delivery_fee"]
            else:
                fee = DeliveryFee(
                    wilaya_code=code,
                    wilaya_name=name,
                    home_delivery_fee=it.get("home_delivery_fee", 0),
                    office_delivery_fee=it.get("office_delivery_fee"),
                )
                db.add(fee)
            upserted += 1
        db.commit()
        return {"success": True, "upserted": upserted}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error receiving delivery fees update: {e}")
        raise HTTPException(status_code=500, detail="Failed to update delivery fees")

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
    # Render requires binding to 0.0.0.0 and the PORT env var
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)