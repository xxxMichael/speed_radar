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

def _send_email_worker(vehicle_id: int, plate: str, speed: float, speed_limit: float, fine_amount: float, frame_crop=None):
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
    msg['Subject'] = f"🚔 ALERTA DE INFRACCIÓN: Vehículo #{vehicle_id} - Placa {plate}"
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
                    <h2>Multa Estimada por Lógica Difusa</h2>
                    <p>El sistema difuso Mamdani ha calculado la severidad basada en el exceso de velocidad y el límite de la vía.</p>
                </div>
                
                <table>
                    <tr>
                        <th>ID de Registro</th>
                        <td>#{vehicle_id}</td>
                    </tr>
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
                        <th>Monto Multa Estimado</th>
                        <td class="fine-value">${fine_amount:.2f} USD</td>
                    </tr>
                </table>

                {"" if frame_crop is None else f'''
                <div class="evidence">
                    <h3 style="margin-top:0; color:#455a64; font-size: 15px;">Evidencia Fotográfica de la Placa</h3>
                    <img src="cid:evidence_image" alt="Captura de placa vehicular" />
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

    # Adjuntar la imagen si está disponible
    if frame_crop is not None:
        try:
            # Codificar imagen cv2 en JPG
            ret, buf = cv2.imencode('.jpg', frame_crop)
            if ret:
                img_data = buf.tobytes()
                mime_img = MIMEImage(img_data, name='evidence.jpg')
                mime_img.add_header('Content-ID', '<evidence_image>')
                mime_img.add_header('Content-Disposition', 'inline', filename='evidence.jpg')
                msg.attach(mime_img)
        except Exception as e:
            print(f"[SMTP WARNING] No se pudo adjuntar la imagen al correo: {e}")

    # Envío mediante smtplib
    try:
        # Usar conexión TLS segura
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, recipient, msg.as_string())
        server.quit()
        print(f"[SMTP] Correo enviado exitosamente para el vehiculo #{vehicle_id} (Placa: {plate}, Multa: ${fine_amount:.2f})")
    except Exception as e:
        print(f"[SMTP ERROR] Falla al enviar el correo: {e}")


def send_infraction_email(vehicle_id: int, plate: str, speed: float, speed_limit: float, fine_amount: float, frame_crop=None):
    """
    Función pública para disparar el correo de alerta.
    Inicia un hilo en segundo plano de manera inmediata para no interferir con el rendimiento del radar.
    """
    print(f"[SMTP] Iniciando envío de correo de alerta para Vehículo #{vehicle_id}...")
    t = threading.Thread(
        target=_send_email_worker,
        args=(vehicle_id, plate, speed, speed_limit, fine_amount, frame_crop),
        daemon=True
    )
    t.start()
