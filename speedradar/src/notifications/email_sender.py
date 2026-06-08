"""
Módulo para el envío de notificaciones automáticas por correo electrónico ante infracciones.
Carga credenciales desde un archivo .env en la raíz del proyecto y envía correos HTML asíncronos.
"""

import os
import smtplib
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import cv2

# Lector simple de archivos .env para evitar dependencias externas adicionales
def load_env():
    # Buscar .env en el directorio actual de ejecución o dos niveles arriba de este script
    paths_to_try = [
        os.path.join(os.getcwd(), '.env'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, val = line.split('=', 1)
                            # Limpiar comillas si existen
                            val = val.strip().strip('"').strip("'")
                            os.environ[key.strip()] = val
                break
            except Exception as e:
                print(f"[SMTP WARNING] No se pudo leer el archivo .env en {path}: {e}")

# Ejecutar carga al importar el módulo
load_env()

def _send_email_worker(vehicle_id: int, plate: str, speed: float, speed_limit: float, sanction_hours: float, vehicle_crop=None, plate_crop=None):
    """
    Función interna ejecutada en un hilo separado para evitar colgar el video en tiempo real.
    """
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port_str = os.getenv("SMTP_PORT")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    recipient = os.getenv("EMAIL_RECIPIENT")

    if not all([smtp_server, smtp_port_str, smtp_user, smtp_password, recipient]):
        print("[SMTP ERROR] Faltan variables de configuración en el archivo .env para el envío de correos.")
        return

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        print(f"[SMTP ERROR] Puerto SMTP inválido: '{smtp_port_str}'. Debe ser un número entero.")
        return

    # Formatear datos
    exceso = speed - speed_limit
    hora_actual = time.strftime("%d/%m/%Y %H:%M:%S")

    # Crear el objeto de mensaje MIME Multipart
    msg = MIMEMultipart('related')
    msg['Subject'] = f"🚔 ALERTA DE INFRACCIÓN: Placa {plate}"
    msg['From'] = smtp_user
    msg['To'] = recipient

    # Cuerpo en HTML con diseño premium
    html_body = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #333333;
                background-color: #f4f6f9;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 600px;
                margin: 20px auto;
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.08);
                overflow: hidden;
                border: 1px solid #e1e8ed;
            }}
            .header {{
                background: linear-gradient(135deg, #d32f2f, #b71c1c);
                color: #ffffff;
                padding: 25px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .header p {{
                margin: 5px 0 0 0;
                opacity: 0.9;
                font-size: 14px;
            }}
            .content {{
                padding: 25px;
            }}
            .alert-box {{
                background-color: #ffebee;
                border-left: 5px solid #d32f2f;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 4px;
            }}
            .alert-box h2 {{
                margin: 0 0 5px 0;
                color: #c62828;
                font-size: 18px;
            }}
            .alert-box p {{
                margin: 0;
                font-size: 14px;
                color: #555555;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 25px;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #eceff1;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: 600;
                color: #546e7a;
                width: 45%;
            }}
            td {{
                color: #263238;
            }}
            .fine-value {{
                font-size: 18px;
                font-weight: bold;
                color: #d32f2f;
            }}
            .evidence {{
                text-align: center;
                margin-top: 15px;
                background-color: #fafafa;
                padding: 15px;
                border-radius: 6px;
                border: 1px dashed #cfd8dc;
            }}
            .evidence img {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                border: 2px solid #b0bec5;
                margin-bottom: 10px;
            }}
            .footer {{
                background-color: #37474f;
                color: #b0bec5;
                text-align: center;
                padding: 15px;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Reporte de Infracción Vial</h1>
                <p>Sistema Inteligente de Control de Velocidad y Radar OCR</p>
            </div>
            <div class="content">
                <div class="alert-box">
                    <h2>Sanción de Ingreso por Lógica Difusa</h2>
                    <p>El sistema difuso Mamdani ha calculado el tiempo de suspensión basado en el exceso de velocidad en el campus universitario.</p>
                </div>
                
                <table>
                    <tr>
                        <th>Placa del Vehículo</th>
                        <td><strong>{plate}</strong></td>
                    </tr>
                    <tr>
                        <th>Velocidad Medida</th>
                        <td>{speed:.1f} km/h</td>
                    </tr>
                    <tr>
                        <th>Límite de la Vía</th>
                        <td>{speed_limit:.1f} km/h</td>
                    </tr>
                    <tr>
                        <th>Exceso Registrado</th>
                        <td>+{exceso:.1f} km/h</td>
                    </tr>
                    <tr>
                        <th>Fecha y Hora</th>
                        <td>{hora_actual}</td>
                    </tr>
                    <tr>
                        <th>Suspensión de Ingreso Sugerida</th>
                        <td class="fine-value">{sanction_hours:.1f} horas</td>
                    </tr>
                </table>

                {"" if vehicle_crop is None else f'''
                <div class="evidence">
                    <h3 style="margin-top:0; color:#455a64; font-size: 15px;">Evidencia del Vehículo</h3>
                    <img src="cid:vehicle_image" alt="Captura general del vehiculo" />
                </div>
                '''}
                
                {"" if plate_crop is None else f'''
                <div class="evidence">
                    <h3 style="margin-top:0; color:#455a64; font-size: 15px;">Detalle de la Placa Analizada</h3>
                    <img src="cid:plate_image" alt="Recorte de placa" />
                </div>
                '''}
            </div>
            <div class="footer">
                Este correo fue enviado de manera automática por el Radar de Velocidad AI.
                <br>No responda a este mensaje.
            </div>
        </div>
    </body>
    </html>
    """

    # Adjuntar HTML
    msg.attach(MIMEText(html_body, 'html'))

    # Adjuntar la imagen del vehiculo si está disponible
    if vehicle_crop is not None:
        try:
            ret, buf = cv2.imencode('.jpg', vehicle_crop)
            if ret:
                img_data = buf.tobytes()
                mime_img = MIMEImage(img_data, name='vehicle.jpg')
                mime_img.add_header('Content-ID', '<vehicle_image>')
                mime_img.add_header('Content-Disposition', 'inline', filename='vehicle.jpg')
                msg.attach(mime_img)
        except Exception as e:
            print(f"[SMTP WARNING] No se pudo adjuntar la imagen del vehículo: {e}")

    # Adjuntar la imagen de la placa si está disponible
    if plate_crop is not None:
        try:
            ret, buf = cv2.imencode('.jpg', plate_crop)
            if ret:
                img_data = buf.tobytes()
                mime_img = MIMEImage(img_data, name='plate.jpg')
                mime_img.add_header('Content-ID', '<plate_image>')
                mime_img.add_header('Content-Disposition', 'inline', filename='plate.jpg')
                msg.attach(mime_img)
        except Exception as e:
            print(f"[SMTP WARNING] No se pudo adjuntar la imagen de la placa: {e}")

    # Envío mediante smtplib
    try:
        # Usar conexión TLS segura
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, recipient, msg.as_string())
        server.quit()
        print(f"[SMTP] Correo enviado exitosamente para el vehiculo #{vehicle_id} (Placa: {plate}, Sanción: {sanction_hours:.1f} h)")
    except Exception as e:
        print(f"[SMTP ERROR] Falla al enviar el correo: {e}")


def send_infraction_email(vehicle_id: int, plate: str, speed: float, speed_limit: float, sanction_hours: float, vehicle_crop=None, plate_crop=None):
    """
    Función pública para disparar el correo de alerta.
    Inicia un hilo en segundo plano de manera inmediata para no interferir con el rendimiento del radar.
    """
    print(f"[SMTP] Iniciando envío de correo de alerta para Vehículo #{vehicle_id}...")
    t = threading.Thread(
        target=_send_email_worker,
        args=(vehicle_id, plate, speed, speed_limit, sanction_hours, vehicle_crop, plate_crop),
        daemon=True
    )
    t.start()


def _send_test_email_worker(plate: str, vehicle_crop=None, plate_crop=None, det_method="Desconocido", det_conf=0.0, avg_char_conf=0.0):
    """
    Worker interno para enviar el correo de prueba OCR sin formato de infracción.
    """
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port_str = os.getenv("SMTP_PORT")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    recipient = os.getenv("EMAIL_RECIPIENT")

    if not all([smtp_server, smtp_port_str, smtp_user, smtp_password, recipient]):
        print("[SMTP ERROR] Faltan variables de configuración en el archivo .env para el envío de correos.")
        return

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        return

    hora_actual = time.strftime("%d/%m/%Y %H:%M:%S")

    msg = MIMEMultipart('related')
    msg['Subject'] = f"📷 Prueba de Detección OCR: Placa {plate}"
    msg['From'] = smtp_user
    msg['To'] = recipient

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; color: #333; }}
            .container {{ max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ccc; border-radius: 8px; }}
            h2 {{ color: #2980b9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Test de Sistema OCR en Tiempo Real</h2>
            <p>Se ha detectado una placa en la cámara web.</p>
            <ul>
                <li><strong>Placa leída:</strong> {plate}</li>
                <li><strong>Fecha y Hora:</strong> {hora_actual}</li>
                <li><strong>Método de Detección:</strong> {det_method}</li>
                <li><strong>Confianza Detección:</strong> {det_conf:.1%}</li>
                <li><strong>Confianza Promedio Letras:</strong> {avg_char_conf:.1%}</li>
            </ul>
            {"" if vehicle_crop is None else '<p><strong>Vehículo:</strong></p><img src="cid:vehicle_image" width="400" />'}
            {"" if plate_crop is None else '<p><strong>Placa (Debug OCR):</strong></p><img src="cid:plate_image" width="200" />'}
        </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(html_body, 'html'))

    if vehicle_crop is not None:
        ret, buf = cv2.imencode('.jpg', vehicle_crop)
        if ret:
            img_data = buf.tobytes()
            mime_img = MIMEImage(img_data, name='vehicle.jpg')
            mime_img.add_header('Content-ID', '<vehicle_image>')
            mime_img.add_header('Content-Disposition', 'inline', filename='vehicle.jpg')
            msg.attach(mime_img)

    if plate_crop is not None:
        ret, buf = cv2.imencode('.jpg', plate_crop)
        if ret:
            img_data = buf.tobytes()
            mime_img = MIMEImage(img_data, name='plate.jpg')
            mime_img.add_header('Content-ID', '<plate_image>')
            mime_img.add_header('Content-Disposition', 'inline', filename='plate.jpg')
            msg.attach(mime_img)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, recipient, msg.as_string())
        server.quit()
        print(f"[SMTP] Correo de prueba enviado para placa: {plate}")
    except Exception as e:
        print(f"[SMTP ERROR] {e}")


def send_ocr_test_email(plate: str, vehicle_crop=None, plate_crop=None, det_method="Desconocido", det_conf=0.0, avg_char_conf=0.0):
    t = threading.Thread(target=_send_test_email_worker, args=(plate, vehicle_crop, plate_crop, det_method, det_conf, avg_char_conf), daemon=True)
    t.start()

