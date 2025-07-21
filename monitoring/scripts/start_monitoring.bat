@echo off
echo 🔍 Starting OpenStack RCA Monitoring Stack (Windows)
echo =======================================================

cd /d "%~dp0\.."

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    echo    Make sure Docker Desktop is running and try again.
    pause
    exit /b 1
)

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ docker-compose not found. Please install docker-compose.
    pause
    exit /b 1
)

echo ✅ Docker is running

REM Create necessary directories
echo 📁 Creating directories...
if not exist "prometheus\rules" mkdir prometheus\rules
if not exist "prometheus\file_sd" mkdir prometheus\file_sd
if not exist "grafana\provisioning\datasources" mkdir grafana\provisioning\datasources
if not exist "grafana\provisioning\dashboards" mkdir grafana\provisioning\dashboards
if not exist "grafana\dashboards" mkdir grafana\dashboards

REM Create file service discovery config
echo 📝 Creating file service discovery config...
echo [> prometheus\file_sd\app_metrics.json
echo   {>> prometheus\file_sd\app_metrics.json
echo     "targets": ["host.docker.internal:8502"],>> prometheus\file_sd\app_metrics.json
echo     "labels": {>> prometheus\file_sd\app_metrics.json
echo       "job": "rca-custom-metrics",>> prometheus\file_sd\app_metrics.json
echo       "service": "rca-application">> prometheus\file_sd\app_metrics.json
echo     }>> prometheus\file_sd\app_metrics.json
echo   }>> prometheus\file_sd\app_metrics.json
echo ]>> prometheus\file_sd\app_metrics.json

REM Start the monitoring stack
echo 🚀 Starting monitoring services...
docker-compose up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 15 /nobreak >nul

REM Check service health
echo 🏥 Checking service health...

REM Check Prometheus
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:9090/-/ready' -Method GET -TimeoutSec 5 | Out-Null; Write-Host '✅ Prometheus is ready (http://localhost:9090)' } catch { Write-Host '⚠️  Prometheus not yet ready' }"

REM Check Grafana  
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:3000/api/health' -Method GET -TimeoutSec 5 | Out-Null; Write-Host '✅ Grafana is ready (http://localhost:3000)'; Write-Host '   Default login: admin/admin123' } catch { Write-Host '⚠️  Grafana not yet ready' }"

REM Check Node Exporter
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:9100/metrics' -Method GET -TimeoutSec 5 | Out-Null; Write-Host '✅ Node Exporter is ready (http://localhost:9100)' } catch { Write-Host '⚠️  Node Exporter not yet ready' }"

REM Check Process Exporter
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:9256/metrics' -Method GET -TimeoutSec 5 | Out-Null; Write-Host '✅ Process Exporter is ready (http://localhost:9256)' } catch { Write-Host '⚠️  Process Exporter not yet ready' }"

echo.
echo 🎉 Monitoring stack started successfully!
echo.
echo 📊 Access URLs:
echo    Grafana:    http://localhost:3000 (admin/admin123)
echo    Prometheus: http://localhost:9090
echo    Node Exp:   http://localhost:9100/metrics
echo    Process Exp: http://localhost:9256/metrics
echo.
echo 🔧 Next steps:
echo 1. Start your RCA application: python main.py --mode streamlit
echo 2. Start custom metrics exporter: python monitoring\scripts\export_app_metrics.py
echo 3. Open Grafana and import the provided dashboards
echo.
echo 🛑 To stop: docker-compose down
echo.
pause