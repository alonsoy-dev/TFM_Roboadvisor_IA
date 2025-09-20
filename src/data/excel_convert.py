import os, time, gc
import psutil
import pythoncom
import win32process
from win32com.client import gencache

"""
Convertidor de xls a xlsx local. Solución de contorno por incompatibilidad con uso de xldr
"""
XL_OPENXML_WORKBOOK = 51  # .xlsx

class ExcelTimeout(Exception): ...
class ExcelConvertError(Exception): ...

def _get_pid_from_hwnd(hwnd: int) -> int | None:
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return pid
    except Exception:
        return None

def _kill_pid(pid: int):
    try:
        p = psutil.Process(pid)
        for ch in p.children(recursive=True):
            ch.kill()
        p.kill()
    except Exception:
        pass

def convert_with_excel(xls_path: str, timeout_sec: int = 15, retries: int = 2) -> str:
    """
    Abre un .xls con Excel (COM) y lo guarda como .xlsx.
    Si Excel se queda bloqueado, mata el proceso y reintenta.
    Devuelve la ruta .xlsx final.
    """
    xls_path = os.path.abspath(xls_path)
    if not xls_path.lower().endswith(".xls"):
        return xls_path

    xlsx_path = os.path.splitext(xls_path)[0] + ".xlsx"

    last_err = None
    for attempt in range(1, retries + 1):
        app = None
        wb = None
        pid = None
        try:
            pythoncom.CoInitialize()
            # Generate cached wrappers (más estable que DispatchEx)
            app = gencache.EnsureDispatch("Excel.Application")
            app.Visible = False
            app.ScreenUpdating = False
            app.DisplayAlerts = False
            app.EnableEvents = False
            try:
                app.AutomationSecurity = 3  # deshabilita macros
            except Exception:
                pass

            # intenta obtener el PID para poder matarlo si se queda colgado
            try:
                hwnd = int(app.Hwnd)
                pid = _get_pid_from_hwnd(hwnd)
            except Exception:
                pid = None

            # Abrir en read-only, sin actualizar vínculos, ignorando recomendaciones
            wb = app.Workbooks.Open(
                xls_path,
                UpdateLinks=0,
                ReadOnly=True,
                IgnoreReadOnlyRecommended=True,
            )

            # Guardar como xlsx
            wb.SaveAs(xlsx_path, FileFormat=XL_OPENXML_WORKBOOK)
            wb.Close(SaveChanges=False)

            # Cerrar Excel limpio
            app.Quit()
            app = None

            # pequeña pausa para liberar handles
            time.sleep(0.2)
            gc.collect()
            return xlsx_path

        except Exception as e:
            last_err = e
            # Intento de cierre limpio
            try:
                if wb is not None:
                    wb.Close(SaveChanges=False)
            except Exception:
                pass
            try:
                if app is not None:
                    app.Quit()
            except Exception:
                pass

            # Si Excel quedó colgado, mata el proceso asociado
            if pid:
                _kill_pid(pid)

            # si no es el último intento, espera un poco y reintenta
            if attempt < retries:
                time.sleep(0.5)
                continue
            else:
                raise ExcelConvertError(f"Fallo al convertir {os.path.basename(xls_path)}: {e}") from e

        finally:
            # Si Excel aún vive, mátalo tras timeout
            start = time.time()
            if pid:
                try:
                    p = psutil.Process(pid)
                    while p.is_running() and (time.time() - start) < timeout_sec:
                        time.sleep(0.1)
                except Exception:
                    pass
                # pasado el timeout, mata
                try:
                    p = psutil.Process(pid)
                    if p.is_running():
                        _kill_pid(pid)
                except Exception:
                    pass
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass
            gc.collect()
    raise ExcelConvertError(f"No se pudo convertir {xls_path}: {last_err}")
