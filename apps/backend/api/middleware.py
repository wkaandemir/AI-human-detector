"""
API Middleware'leri - Rate Limiting, Request Logging, vb.
"""

import time
import logging
from typing import Dict, Callable, Optional
from collections import defaultdict
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from datetime import datetime, timedelta
import hashlib


# Logger yapılandırması
logger = logging.getLogger("api.middleware")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.

    IP adresine göre istek sınırlaması uygular.
    Sliding window algorithm kullanır.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        trusted_proxies: Optional[list] = None
    ):
        """
        Rate limiting middleware yapıcısı.

        Args:
            app: ASGI uygulaması
            requests_per_minute: Dakikadaki maksimum istek sayısı
            requests_per_hour: Saatteki maksimum istek sayısı
            trusted_proxies: Güvenilir proxy IP'leri (X-Forwarded-For için)
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.trusted_proxies = trusted_proxies or []

        # IP bazlı istek takibi
        # {ip: {"minute": [(timestamp, ...)], "hour": [(timestamp, ...)]}}
        self.request_history: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {"minute": [], "hour": []}
        )

        # Cleanup timer
        self._last_cleanup = time.time()

    def _get_client_ip(self, request: Request) -> str:
        """
        İstemci IP adresini alır.

        X-Forwarded-For header'ını kontrol eder (proxy arkasında çalışma için).

        Args:
            request: FastAPI request objesi

        Returns:
            İstemci IP adresi
        """
        # X-Forwarded-For header'ını kontrol et
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Virgülle ayrılmış IP listesi, ilk IP gerçek istemci
            ips = [ip.strip() for ip in forwarded.split(",")]
            return ips[0]

        # X-Real-IP header'ını kontrol et
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Direct connection
        if request.client:
            return request.client.host

        return "unknown"

    def _is_rate_limited(self, ip: str) -> tuple[bool, Optional[str]]:
        """
        IP adresinin rate limit'e takılıp takılmadığını kontrol eder.

        Args:
            ip: İstemci IP adresi

        Returns:
            (limit_exceeded, retry_after) tuple
        """
        now = time.time()
        now_dt = datetime.now()

        # Periodik cleanup (her dakika)
        if now - self._last_cleanup > 60:
            self._cleanup_old_entries(now)
            self._last_cleanup = now

        history = self.request_history[ip]

        # Dakikalık limit kontrolü
        minute_ago = now - 60
        history["minute"] = [ts for ts in history["minute"] if ts > minute_ago]

        if len(history["minute"]) >= self.requests_per_minute:
            # En eski isteğin zamanını bul
            oldest_request = min(history["minute"])
            retry_after = int(60 - (now - oldest_request)) + 1
            return True, str(retry_after)

        # Saatlik limit kontrolü
        hour_ago = now - 3600
        history["hour"] = [ts for ts in history["hour"] if ts > hour_ago]

        if len(history["hour"]) >= self.requests_per_hour:
            oldest_request = min(history["hour"])
            retry_after = int(3600 - (now - oldest_request)) + 1
            return True, str(retry_after)

        return False, None

    def _cleanup_old_entries(self, now: float):
        """Eski istek kayıtlarını temizler"""
        hour_ago = now - 3600

        for ip in list(self.request_history.keys()):
            history = self.request_history[ip]

            # Dakika ve saat verilerini temizle
            history["minute"] = [ts for ts in history["minute"] if ts > hour_ago]
            history["hour"] = [ts for ts in history["hour"] if ts > hour_ago]

            # Boş entry'leri sil
            if not history["minute"] and not history["hour"]:
                del self.request_history[ip]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Rate limiting kontrolü yapar ve isteği işler.

        Args:
            request: Gelen istek
            call_next: Sonraki middleware/endpoint

        Returns:
            HTTP Response veya 429 Too Many Requests
        """
        # Health check ve dokümantasyon endpoint'lerini rate limit'e dahil etme
        if request.url.path in ["/", "/health", "/ping", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # İstemci IP'sini al
        client_ip = self._get_client_ip(request)

        # Rate limit kontrolü
        is_limited, retry_after = self._is_rate_limited(client_ip)

        if is_limited:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "message": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": retry_after,
                    "X-RateLimit-Limit": f"{self.requests_per_minute}/minute, {self.requests_per_hour}/hour"
                }
            )

        # İsteği kaydet
        now = time.time()
        self.request_history[client_ip]["minute"].append(now)
        self.request_history[client_ip]["hour"].append(now)

        # Rate limit header'larını ekle
        response = await call_next(request)

        minute_count = len(self.request_history[client_ip]["minute"])
        hour_count = len(self.request_history[client_ip]["hour"])

        response.headers["X-RateLimit-Limit"] = f"{self.requests_per_minute}/minute, {self.requests_per_hour}/hour"
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - minute_count)
        )
        response.headers["X-RateLimit-Reset"] = str(int(now + 60))

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware.

    Gelen istekleri log'lar (response time, status code, vb.).
    """

    def __init__(
        self,
        app: ASGIApp,
        log_level: int = logging.INFO,
        log_body: bool = False,
        max_body_size: int = 1000
    ):
        """
        Request logging middleware yapıcısı.

        Args:
            app: ASGI uygulaması
            log_level: Log seviyesi
            log_body: Request body'sini log'la ( güvenlik riski olabilir)
            max_body_size: Log'lanacak maksimum body boyutu
        """
        super().__init__(app)
        self.log_level = log_level
        self.log_body = log_body
        self.max_body_size = max_body_size

        # Logger yapılandırması
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        İsteği log'lar ve işler.

        Args:
            request: Gelen istek
            call_next: Sonraki middleware/endpoint

        Returns:
            HTTP Response
        """
        start_time = time.time()

        # Request bilgileri
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""

        # User-Agent
        user_agent = request.headers.get("User-Agent", "unknown")

        # Request ID oluştur
        request_id = hashlib.md5(
            f"{client_ip}-{start_time}".encode()
        ).hexdigest()[:8]

        # Request ID'yi state'e ekle
        request.state.request_id = request_id

        # Body log'lama (opsiyonel)
        body_info = ""
        if self.log_body and method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    body_str = body.decode("utf-8", errors="ignore")
                    if len(body_str) > self.max_body_size:
                        body_str = body_str[:self.max_body_size] + "... (truncated)"
                    body_info = f" | Body: {body_str}"
            except Exception:
                body_info = " | Body: <unreadable>"

        # Log başlangıcı
        logger.log(
            self.log_level,
            f"[{request_id}] {method} {path}?{query_params} from {client_ip} | UA: {user_agent}{body_info}"
        )

        # İsteği işle
        try:
            response = await call_next(request)

            # Response süresi
            process_time = time.time() - start_time

            # Response header'larına timing ekle
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            response.headers["X-Request-ID"] = request_id

            # Log bitişi
            logger.log(
                self.log_level,
                f"[{request_id}] {method} {path} | Status: {response.status_code} | Time: {process_time:.3f}s"
            )

            return response

        except Exception as e:
            # Hata log'lama
            process_time = time.time() - start_time

            logger.error(
                f"[{request_id}] {method} {path} | Error: {str(e)} | Time: {process_time:.3f}s"
            )

            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Güvenlik header'larını ekleyen middleware.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Güvenlik header'larını ekler ve isteği işler.

        Args:
            request: Gelen istek
            call_next: Sonraki middleware/endpoint

        Returns:
            HTTP Response
        """
        response = await call_next(request)

        # Güvenlik header'ları
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """
    Health check için özel middleware.

    Health check endpoint'lerini hızlıca yanıtlar (rate limiting ve logging bypass).
    """

    def __init__(self, app: ASGIApp, health_paths: Optional[list] = None):
        """
        Health check middleware yapıcısı.

        Args:
            app: ASGI uygulaması
            health_paths: Health check path'leri
        """
        super().__init__(app)
        self.health_paths = health_paths or ["/health", "/ping", "/readiness"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Health check için hızlı yanıt.

        Args:
            request: Gelen istek
            call_next: Sonraki middleware/endpoint

        Returns:
            HTTP Response
        """
        # Health check endpoint'lerini hızlıca yanıtla
        if request.url.path in self.health_paths:
            return await call_next(request)

        return await call_next(request)


# FastAPI'nin JSONResponse'unu import et
from fastapi.responses import JSONResponse
