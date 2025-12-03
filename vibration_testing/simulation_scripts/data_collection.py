import machine
import utime
from machine import Pin, I2C
import ustruct

class PCA9548A:
    """I2C Multiplexer for managing multiple devices with same address"""
    def __init__(self, i2c, addr=0x70):
        self.i2c = i2c
        self.addr = addr
        self.current_channel = None
        
    def select_channel(self, channel):
        """Select a channel (0-7) on the multiplexer"""
        if channel < 0 or channel > 7:
            raise ValueError("Channel must be 0-7")
        
        # Write channel selection byte (bit pattern)
        channel_byte = 1 << channel
        self.i2c.writeto(self.addr, bytes([channel_byte]))
        self.current_channel = channel
        utime.sleep_ms(1)  # Small delay for channel switching
        
    def disable_all_channels(self):
        """Disable all channels"""
        self.i2c.writeto(self.addr, bytes([0x00]))
        self.current_channel = None
        utime.sleep_ms(1)

class MPU6050_Direct:
    """MPU6050 class for direct I2C connection (no PCA multiplexer)"""
    def __init__(self, i2c, addr=0x68, name="MPU"):
        self.i2c = i2c
        self.addr = addr
        self.name = name
        
        # Calibration offsets
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        
        # Wake up the MPU6050
        try:
            self.i2c.writeto_mem(self.addr, 0x6B, b'\x00')
            utime.sleep_ms(100)
            print(f"{self.name} direct connection (0x{self.addr:02X}) initialized successfully")
        except Exception as e:
            print(f"Failed to initialize {self.name} at 0x{self.addr:02X}: {e}")
            raise
    
    def read_raw_data(self, addr):
        data = self.i2c.readfrom_mem(self.addr, addr, 2)
        value = ustruct.unpack('>h', data)[0]
        return value
    
    def get_raw_acceleration(self):
        """Get raw acceleration without calibration"""
        accel_x_raw = self.read_raw_data(0x3B)
        accel_y_raw = self.read_raw_data(0x3D)
        accel_z_raw = self.read_raw_data(0x3F)
        
        # Convert to g's
        accel_x = accel_x_raw / 16384.0
        accel_y = accel_y_raw / 16384.0
        accel_z = accel_z_raw / 16384.0
        
        return accel_x, accel_y, accel_z
    
    def get_acceleration(self):
        """Get calibrated acceleration"""
        accel_x, accel_y, accel_z = self.get_raw_acceleration()
        
        # Apply calibration offsets
        accel_x -= self.offset_x
        accel_y -= self.offset_y
        accel_z -= self.offset_z
        
        return accel_x, accel_y, accel_z
    
    def calibrate(self, samples=200):
        """Calibrate accelerometer - sensor must be level and stationary"""
        print(f"Calibrating {self.name}... Keep sensor level and still for {samples//10} seconds")
        
        sum_x, sum_y, sum_z = 0, 0, 0
        
        for i in range(samples):
            try:
                accel_x, accel_y, accel_z = self.get_raw_acceleration()
                sum_x += accel_x
                sum_y += accel_y
                sum_z += accel_z
                
                if i % 20 == 0:  # Progress indicator
                    print(".", end="")
                
                utime.sleep_ms(10)
            except Exception as e:
                print(f"\nError during calibration of {self.name}: {e}")
                return None, None, None
        
        # Calculate average readings
        avg_x = sum_x / samples
        avg_y = sum_y / samples
        avg_z = sum_z / samples
        
        # Set offsets (Z should read +1g when level, X and Y should read 0g)
        self.offset_x = avg_x
        self.offset_y = avg_y
        self.offset_z = avg_z - 1.0  # Subtract 1g for gravity
        
        print(f"\n{self.name} calibration complete!")
        print(f"Offsets: X={self.offset_x:.4f}, Y={self.offset_y:.4f}, Z={self.offset_z:.4f}")

class MPU6050_PCA:
    """MPU6050 class for connection through PCA9548A multiplexer"""
    def __init__(self, i2c, pca, channel, addr=0x68, name="MPU"):
        self.i2c = i2c
        self.pca = pca
        self.channel = channel
        self.addr = addr
        self.name = name
        
        # Calibration offsets
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        
        # Initialize the sensor
        self._select_and_init()
        
    def _select_and_init(self):
        """Select PCA channel and initialize MPU6050"""
        try:
            self.pca.select_channel(self.channel)
            utime.sleep_ms(10)
            # Wake up the MPU6050
            self.i2c.writeto_mem(self.addr, 0x6B, b'\x00')
            utime.sleep_ms(100)
            print(f"{self.name} on PCA channel {self.channel} (0x{self.addr:02X}) initialized successfully")
        except Exception as e:
            print(f"Failed to initialize {self.name} on channel {self.channel}: {e}")
            raise
    
    def read_raw_data(self, addr):
        """Read raw data from sensor (with PCA channel selection)"""
        self.pca.select_channel(self.channel)
        utime.sleep_ms(2)  # Small delay for channel switching
        data = self.i2c.readfrom_mem(self.addr, addr, 2)
        value = ustruct.unpack('>h', data)[0]
        return value
    
    def get_raw_acceleration(self):
        """Get raw acceleration without calibration"""
        accel_x_raw = self.read_raw_data(0x3B)
        accel_y_raw = self.read_raw_data(0x3D)
        accel_z_raw = self.read_raw_data(0x3F)
        
        # Convert to g's
        accel_x = accel_x_raw / 16384.0
        accel_y = accel_y_raw / 16384.0
        accel_z = accel_z_raw / 16384.0
        
        return accel_x, accel_y, accel_z
    
    def get_acceleration(self):
        """Get calibrated acceleration"""
        accel_x, accel_y, accel_z = self.get_raw_acceleration()
        
        # Apply calibration offsets
        accel_x -= self.offset_x
        accel_y -= self.offset_y
        accel_z -= self.offset_z
        
        return accel_x, accel_y, accel_z
    
    def calibrate(self, samples=200):
        """Calibrate accelerometer - sensor must be level and stationary"""
        print(f"Calibrating {self.name} on channel {self.channel}... Keep sensor level and still for {samples//10} seconds")
        
        sum_x, sum_y, sum_z = 0, 0, 0
        
        for i in range(samples):
            try:
                accel_x, accel_y, accel_z = self.get_raw_acceleration()
                sum_x += accel_x
                sum_y += accel_y
                sum_z += accel_z
                
                if i % 20 == 0:  # Progress indicator
                    print(".", end="")
                
                utime.sleep_ms(10)
            except Exception as e:
                print(f"\nError during calibration of {self.name}: {e}")
                return None, None, None
        
        # Calculate average readings
        avg_x = sum_x / samples
        avg_y = sum_y / samples
        avg_z = sum_z / samples
        
        # Set offsets (Z should read +1g when level, X and Y should read 0g)
        self.offset_x = avg_x
        self.offset_y = avg_y
        self.offset_z = avg_z - 1.0  # Subtract 1g for gravity
        
        print(f"\n{self.name} calibration complete!")
        print(f"Offsets: X={self.offset_x:.4f}, Y={self.offset_y:.4f}, Z={self.offset_z:.4f}")
        return self.offset_x, self.offset_y, self.offset_z

def wait_for_sensors_to_settle(sensors, settling_time_sec=5, stability_threshold=0.05, check_samples=20):
    """
    Wait for all sensors to settle before starting data recording.
    
    Args:
        sensors: List of sensor objects
        settling_time_sec: Minimum time to wait for initial settling
        stability_threshold: Maximum standard deviation allowed for stability (in g's)
        check_samples: Number of samples to check for stability
    
    Returns:
        True if all sensors settled successfully
    """
    print(f"\nWaiting for sensors to settle...")
    print(f"Initial settling period: {settling_time_sec} seconds")
    
    # Initial settling time - let sensors warm up and stabilize
    for i in range(settling_time_sec):
        print(f"Initial settling: {i+1}/{settling_time_sec} seconds", end="\r")
        utime.sleep(1)
    
    print("\nChecking sensor stability...")
    
    max_attempts = 10  # Maximum number of stability check attempts
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        print(f"Stability check attempt {attempt}/{max_attempts}")
        
        all_stable = True
        
        for sensor in sensors:
            # Collect samples for stability analysis
            readings = []
            print(f"  Checking {sensor.name}...", end="")
            
            try:
                for _ in range(check_samples):
                    accel_x, accel_y, accel_z = sensor.get_acceleration()
                    magnitude = (accel_x**2 + accel_y**2 + accel_z**2)**0.5
                    readings.append((accel_x, accel_y, accel_z, magnitude))
                    utime.sleep_ms(50)  # 20Hz sampling for stability check
                
                # Calculate standard deviations for each axis and magnitude
                if readings:
                    x_vals = [r[0] for r in readings]
                    y_vals = [r[1] for r in readings]
                    z_vals = [r[2] for r in readings]
                    mag_vals = [r[3] for r in readings]
                    
                    # Calculate means
                    x_mean = sum(x_vals) / len(x_vals)
                    y_mean = sum(y_vals) / len(y_vals)
                    z_mean = sum(z_vals) / len(z_vals)
                    mag_mean = sum(mag_vals) / len(mag_vals)
                    
                    # Calculate standard deviations
                    x_std = (sum((x - x_mean)**2 for x in x_vals) / len(x_vals))**0.5
                    y_std = (sum((y - y_mean)**2 for y in y_vals) / len(y_vals))**0.5
                    z_std = (sum((z - z_mean)**2 for z in z_vals) / len(z_vals))**0.5
                    mag_std = (sum((m - mag_mean)**2 for m in mag_vals) / len(mag_vals))**0.5
                    
                    # Check if all axes are stable
                    max_std = max(x_std, y_std, z_std, mag_std)
                    
                    if max_std <= stability_threshold:
                        print(f" STABLE (max_std: {max_std:.4f})")
                    else:
                        print(f" UNSTABLE (max_std: {max_std:.4f}, threshold: {stability_threshold})")
                        all_stable = False
                else:
                    print(" ERROR: No readings collected")
                    all_stable = False
                    
            except Exception as e:
                print(f" ERROR: {e}")
                all_stable = False
        
        if all_stable:
            print("✓ All sensors are stable and ready for data collection!")
            return True
        else:
            print(f"✗ Some sensors are still settling. Waiting 2 seconds before next check...")
            utime.sleep(2)
    
    print(f"⚠ Warning: Not all sensors achieved stability after {max_attempts} attempts.")
    print("Proceeding with data collection anyway. Data quality may be affected.")
    return False

# Configuration
SAMPLE_RATE_HZ = 100
LIVE_MODE = False  # Set to False for file logging, True for live display
CALIBRATE_ON_START = True  # Set to True to calibrate before each session
LOG_DURATION_SEC = 120  # How long to record in seconds (0 = infinite until Ctrl+C)

# Settling configuration
SETTLING_TIME_SEC = 5  # Initial settling time in seconds
STABILITY_THRESHOLD = 0.05  # Maximum allowed standard deviation for stability (g's)
STABILITY_CHECK_SAMPLES = 20  # Number of samples to check for stability

# PCA9548A and sensor configuration
PCA_ADDRESS = 0x70  # Default PCA9548A address
MPU_ADDRESS = 0x68  # MPU6050 address (same for all since using PCA)

# Sensor configuration
# Direct connection (no PCA channel)
DIRECT_SENSORS = [
    {"address": 0x68, "name": "MPU_Direct"},  # Direct connection to Pi
    # Add more direct sensors with different addresses if needed
]

# PCA channel connections
PCA_SENSORS = [
    {"channel": 0, "name": "MPU_Ch0"},  # Sensor on PCA channel 0
    {"channel": 1, "name": "MPU_Ch1"},  # Sensor on PCA channel 1
    # Add more PCA sensors here if needed:
    # {"channel": 2, "name": "MPU_Ch2"},
]

def diagnose_setup(i2c, pca_addr=0x70):
    """Diagnostic function to understand the I2C setup"""
    print("=== I2C DIAGNOSTIC ===")
    
    # Scan main bus
    main_devices = i2c.scan()
    print(f"Main I2C bus devices: {[hex(addr) for addr in main_devices]}")
    
    if pca_addr in main_devices:
        print(f"PCA9548A detected at 0x{pca_addr:02X}")
        
        try:
            # Initialize PCA
            pca = PCA9548A(i2c, pca_addr)
            
            # Test each channel individually
            print("Testing individual PCA channels:")
            for ch in [0, 1]:  # Only test your configured channels
                print(f"\n--- Channel {ch} ---")
                try:
                    pca.select_channel(ch)
                    utime.sleep_ms(50)
                    devices = i2c.scan()
                    print(f"  Devices: {[hex(addr) for addr in devices]}")
                    
                    # Try to communicate with MPU6050 if found
                    if 0x68 in devices:
                        try:
                            # Test basic communication
                            who_am_i = i2c.readfrom_mem(0x68, 0x75, 1)[0]
                            print(f"  MPU6050 WHO_AM_I: 0x{who_am_i:02X} {'(OK)' if who_am_i == 0x68 else '(ERROR)'}")
                        except Exception as e:
                            print(f"  MPU6050 communication failed: {e}")
                            
                except Exception as e:
                    print(f"  Channel {ch} error: {e}")
            
            # Disable all channels and test main bus again
            print(f"\n--- All PCA channels disabled ---")
            pca.disable_all_channels()
            utime.sleep_ms(50)
            devices = i2c.scan()
            print(f"  Main bus devices: {[hex(addr) for addr in devices]}")
            
            if 0x68 in devices:
                print("  Direct MPU6050 detected on main bus!")
                try:
                    who_am_i = i2c.readfrom_mem(0x68, 0x75, 1)[0]
                    print(f"  Direct MPU6050 WHO_AM_I: 0x{who_am_i:02X} {'(OK)' if who_am_i == 0x68 else '(ERROR)'}")
                except Exception as e:
                    print(f"  Direct MPU6050 communication failed: {e}")
            else:
                print("  No direct MPU6050 on main bus")
                
        except Exception as e:
            print(f"PCA initialization failed: {e}")
    else:
        print("No PCA9548A found")
        if 0x68 in main_devices:
            print("Direct MPU6050 found on main bus")

def scan_pca_channels(i2c, pca):
    """Scan all PCA channels to see which have devices"""
    print("Scanning PCA9548A channels...")
    found_channels = []
    
    # First disable all channels to get a clean start
    pca.disable_all_channels()
    utime.sleep_ms(50)
    
    for channel in range(8):
        try:
            # Select specific channel
            pca.select_channel(channel)
            utime.sleep_ms(50)  # More time for channel switching
            
            # Scan for devices on this channel
            devices = i2c.scan()
            
            # Filter out the PCA itself and any devices that appear on main bus
            # Only count devices that are specifically on this channel
            mpu_devices = []
            for addr in devices:
                if addr != pca.addr and addr == 0x68:  # Looking specifically for MPU6050
                    mpu_devices.append(addr)
            
            if mpu_devices:
                print(f"Channel {channel}: Found MPU devices at {[hex(addr) for addr in mpu_devices]}")
                found_channels.append(channel)
            else:
                print(f"Channel {channel}: No MPU devices")
                
        except Exception as e:
            print(f"Channel {channel}: Error - {e}")
    
    # Disable all channels when done
    pca.disable_all_channels()
    utime.sleep_ms(10)
    return found_channels

def main():
    # Try multiple I2C pin configurations for Pico WH
    i2c_configs = [
        (0, 4, 5, "GP4(SDA)/GP5(SCL) - QT Connector"),
        (1, 2, 3, "GP2(SDA)/GP3(SCL) - Standard"),
        (0, 0, 1, "GP0(SDA)/GP1(SCL) - Alternative"),
        (1, 6, 7, "GP6(SDA)/GP7(SCL) - Alternative"),
        (0, 8, 9, "GP8(SDA)/GP9(SCL) - Alternative"),
        (1, 10, 11, "GP10(SDA)/GP11(SCL) - Alternative")
    ]
    
    i2c = None
    for bus_id, sda_pin, scl_pin, description in i2c_configs:
        try:
            print(f"Trying I2C{bus_id} on {description}...")
            i2c = I2C(bus_id, sda=Pin(sda_pin), scl=Pin(scl_pin), freq=400000)
            utime.sleep_ms(100)
            
            # Test if any devices are found on this configuration
            devices = i2c.scan()
            if devices:
                print(f"SUCCESS: Found devices on I2C{bus_id} {description}")
                print(f"Device addresses: {[hex(addr) for addr in devices]}")
                break
            else:
                print(f"No devices found on this configuration")
                
        except Exception as e:
            print(f"Failed to initialize I2C{bus_id}: {e}")
    
    if not i2c:
        print("\nFailed to find working I2C configuration.")
        print("Troubleshooting steps:")
        print("1. Check all connections (direct and PCA)")
        print("2. Verify power to all sensors")
        print("3. Check QT cable connections")
        return
    
    # Get all devices on the main I2C bus (direct connections and PCA)
    main_devices = i2c.scan()
    print(f"Devices on main I2C bus: {[hex(addr) for addr in main_devices]}")
    
    # Run diagnostic to understand the setup better
    diagnose_setup(i2c, PCA_ADDRESS)
    
    # Initialize PCA9548A multiplexer if present
    pca = None
    if PCA_ADDRESS in main_devices:
        try:
            pca = PCA9548A(i2c, PCA_ADDRESS)
            print(f"PCA9548A multiplexer initialized at 0x{PCA_ADDRESS:02X}")
            
            # Since the diagnostic shows channels 0 and 1 work, let's just use those
            found_channels = [0, 1]  # Based on your diagnostic output
            print(f"Using PCA channels: {found_channels}")
            
        except Exception as e:
            print(f"Failed to initialize PCA9548A: {e}")
            return
    else:
        print(f"No PCA9548A found at 0x{PCA_ADDRESS:02X}")
        found_channels = []
    
    # Initialize direct-connected sensors (disable PCA first to avoid conflicts)
    if pca:
        pca.disable_all_channels()
        utime.sleep_ms(50)
    
    sensors = []
    
    # Test if direct sensor is really available when PCA is disabled
    direct_test_devices = i2c.scan()
    print(f"Devices available with PCA disabled: {[hex(addr) for addr in direct_test_devices]}")
    
    for config in DIRECT_SENSORS:
        addr = config["address"]
        name = config["name"]
        
        if addr in direct_test_devices and addr != PCA_ADDRESS:
            try:
                sensor = MPU6050_Direct(i2c, addr, name)
                sensors.append(sensor)
                print(f"Successfully initialized direct sensor: {name}")
            except Exception as e:
                print(f"Failed to initialize direct sensor at 0x{addr:02X}: {e}")
                import sys
                sys.print_exception(e)  # Print full traceback
        else:
            print(f"Direct sensor at 0x{addr:02X} not available (found: {[hex(a) for a in direct_test_devices]})")
    
    # Initialize PCA-connected sensors (only use channels that actually have devices)
    if pca and found_channels:
        print(f"Initializing sensors on PCA channels: {found_channels}")
        for config in PCA_SENSORS:
            channel = config["channel"]
            name = config["name"]
            
            if channel in found_channels:
                try:
                    sensor = MPU6050_PCA(i2c, pca, channel, MPU_ADDRESS, name)
                    sensors.append(sensor)
                    print(f"Successfully initialized PCA sensor: {name} on channel {channel}")
                except Exception as e:
                    print(f"Failed to initialize PCA sensor on channel {channel}: {e}")
                    import sys
                    sys.print_exception(e)  # Print full traceback
            else:
                print(f"No device found on PCA channel {channel} for {name} (available channels: {found_channels})")
    
    if not sensors:
        print("No sensors could be initialized")
        return
    
    print(f"Successfully initialized {len(sensors)} sensors")
    
    # Calibrate sensors if requested
    if CALIBRATE_ON_START:
        print("Place ALL sensors on flat, level surface and press Enter to calibrate...")
        input()  # Wait for user
        for sensor in sensors:
            sensor.calibrate()
        print("All sensors calibrated!\n")
    
    # Wait for sensors to settle before starting data collection
    print("Waiting for sensors to settle before starting data collection...")
    settling_success = wait_for_sensors_to_settle(
        sensors, 
        SETTLING_TIME_SEC, 
        STABILITY_THRESHOLD, 
        STABILITY_CHECK_SAMPLES
    )
    
    if not settling_success:
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            print("Data collection cancelled.")
            return
    
    if LIVE_MODE:
        print("Live acceleration data from all sensors (Press Ctrl+C to stop):")
        try:
            while True:
                for sensor in sensors:
                    try:
                        accel_x, accel_y, accel_z = sensor.get_acceleration()
                        magnitude = (accel_x**2 + accel_y**2 + accel_z**2)**0.5
                        print(f"{sensor.name}: X:{accel_x:+6.3f} Y:{accel_y:+6.3f} Z:{accel_z:+6.3f} Mag:{magnitude:5.3f}")
                    except Exception as e:
                        print(f"{sensor.name}: Error reading data - {e}")
                
                print("-" * 70)  # Separator line
                utime.sleep_ms(int(1000 / SAMPLE_RATE_HZ))
                
        except KeyboardInterrupt:
            print("Stopped")
    
    else:
        # File logging mode
        filename = "pca_multi_accel_data.csv"
        print(f"Starting data logging from {len(sensors)} sensors to {filename}...")
        if LOG_DURATION_SEC > 0:
            print(f"Duration: {LOG_DURATION_SEC} seconds")
        else:
            print("Duration: Until Ctrl+C is pressed")
        
        with open(filename, 'w') as f:
            # Create CSV header for multiple sensors
            header = "timestamp"
            for sensor in sensors:
                header += f",{sensor.name}_x,{sensor.name}_y,{sensor.name}_z,{sensor.name}_mag"
            f.write(header + "\n")
            
            start_time = utime.ticks_ms()
            sample_count = 0
            
            print("Data logging started!")
            
            while True:
                try:
                    elapsed = utime.ticks_diff(utime.ticks_ms(), start_time) / 1000.0
                    
                    # Stop if duration limit reached
                    if LOG_DURATION_SEC > 0 and elapsed >= LOG_DURATION_SEC:
                        break
                    
                    # Build data row
                    row = f"{elapsed:.3f}"
                    all_sensors_ok = True
                    
                    for sensor in sensors:
                        try:
                            accel_x, accel_y, accel_z = sensor.get_acceleration()
                            magnitude = (accel_x**2 + accel_y**2 + accel_z**2)**0.5
                            row += f",{accel_x:.6f},{accel_y:.6f},{accel_z:.6f},{magnitude:.6f}"
                        except Exception as e:
                            print(f"Error reading {sensor.name}: {e}")
                            row += ",NaN,NaN,NaN,NaN"
                            all_sensors_ok = False
                    
                    f.write(row + "\n")
                    sample_count += 1
                    
                    # Progress update every second
                    if sample_count % SAMPLE_RATE_HZ == 0:
                        status = "OK" if all_sensors_ok else "ERRORS"
                        print(f"Time: {elapsed:.1f}s, Samples: {sample_count}, Status: {status}")
                    
                    utime.sleep_ms(int(1000 / SAMPLE_RATE_HZ))
                    
                except KeyboardInterrupt:
                    print(f"\nLogging stopped by user at {elapsed:.1f}s")
                    break
        
        print(f"Data saved to {filename} ({sample_count} samples from {len(sensors)} sensors)")

if __name__ == "__main__":
    main()
