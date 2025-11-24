#!/usr/bin/env python3
"""
Jubilee Y-Axis Movement Test - 3 Cycles for Acceleration Measurements

This script moves the Jubilee tool head across the Y-axis 3 times back and forth.
The tool head will move from min to max, then max to min, repeating 3 cycles total.

Usage:
    python jubilee_y_axis_test_3cycles.py

Make sure to update the JUBILEE_IP variable with your Jubilee's IP address.
"""

import requests
import json
import time
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JubileeYAxisController:
    """
    Controller class for Jubilee Y-axis movement testing with multiple cycles
    """
    
    def __init__(self, ip_address: str, timeout: float = 30.0):
        """
        Initialize the Jubilee controller
        
        :param ip_address: IP address of the Jubilee web controller
        :param timeout: Timeout for HTTP requests in seconds
        """
        self.base_url = f"http://{ip_address}"
        self.timeout = timeout
        self.session = requests.Session()
        
        # Default movement parameters
        self.default_speed = 3000  # mm/min
        
        # Y-axis limits (based on your Jubilee config)
        self.y_min = -34   # mm - safe minimum with margin
        self.y_max = 280   # mm - safe maximum with margin
        
        logger.info(f"Initialized Jubilee Y-axis controller for {self.base_url}")
        logger.info(f"Y-axis range: {self.y_min}mm to {self.y_max}mm")
    
    def send_gcode(self, gcode: str) -> Dict:
        """
        Send G-code command to the Jubilee
        
        :param gcode: G-code command string
        :return: Response from the Jubilee
        """
        try:
            url = f"{self.base_url}/rr_gcode"
            params = {'gcode': gcode}
            
            logger.debug(f"Sending G-code: {gcode}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            return response.json() if response.content else {}
            
        except requests.RequestException as e:
            logger.error(f"Error sending G-code '{gcode}': {e}")
            raise
    
    def get_current_y_position(self) -> float:
        """
        Get current Y position
        
        :return: Current Y position in mm
        """
        try:
            url = f"{self.base_url}/rr_status"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            status = response.json()
            
            if 'coords' in status and 'xyz' in status['coords']:
                return status['coords']['xyz'][1]  # Y is the second coordinate
            else:
                logger.warning("Could not parse Y position from status")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting Y position: {e}")
            return 0.0
    
    def move_y_to(self, y_position: float, speed: Optional[float] = None) -> None:
        """
        Move to specified Y coordinate
        
        :param y_position: Y coordinate in mm
        :param speed: Movement speed in mm/min (None to use default)
        """
        # Check limits
        if not (self.y_min <= y_position <= self.y_max):
            raise ValueError(f"Y position {y_position} outside safe limits ({self.y_min}-{self.y_max}mm)")
        
        # Build G-code command
        move_speed = speed if speed is not None else self.default_speed
        gcode = f"G1 Y{y_position:.2f} F{move_speed:.0f}"
        
        logger.info(f"Moving Y-axis to {y_position}mm at {move_speed}mm/min")
        self.send_gcode(gcode)
    
    def wait_for_movement_complete(self, check_interval: float = 0.5) -> None:
        """
        Wait until movement is complete
        
        :param check_interval: Time between status checks in seconds
        """
        logger.debug("Waiting for Y-axis movement to complete...")
        self.send_gcode("M400")  # Wait for current moves to finish
        time.sleep(check_interval)
        
        # Additional wait for settlement
        timeout_counter = 0
        max_timeout = 60
        
        while timeout_counter < max_timeout:
            try:
                url = f"{self.base_url}/rr_status"
                response = self.session.get(url, timeout=self.timeout)
                status = response.json()
                
                if 'status' in status and status['status'] in ['idle', 'I']:
                    logger.debug("Y-axis movement completed")
                    return
                    
            except Exception as e:
                logger.warning(f"Status check failed: {e}")
            
            time.sleep(check_interval)
            timeout_counter += check_interval
        
        logger.warning("Movement completion check timed out")
    
    def run_y_axis_test_3_cycles(self, pause_time: float = 10.0, speed: Optional[float] = None, cycles: int = 3) -> None:
        """
        Run the Y-axis movement test for multiple cycles
        
        :param pause_time: Time to pause at each position in seconds
        :param speed: Movement speed in mm/min (None for default)
        :param cycles: Number of back-and-forth cycles to perform
        """
        logger.info("="*60)
        logger.info(f"STARTING Y-AXIS MOVEMENT TEST - {cycles} CYCLES")
        logger.info("="*60)
        
        # Get starting position
        start_y = self.get_current_y_position()
        logger.info(f"Starting Y position: {start_y:.2f}mm")
        logger.info(f"Will perform {cycles} complete back-and-forth cycles")
        
        try:
            for cycle in range(1, cycles + 1):
                logger.info(f"\n{'='*20} CYCLE {cycle} of {cycles} {'='*20}")
                
                # Move to minimum Y position
                logger.info(f"Cycle {cycle}: Moving to Y minimum position ({self.y_min}mm)")
                self.move_y_to(self.y_min, speed)
                self.wait_for_movement_complete()
                
                logger.info(f"Cycle {cycle}: Pausing for {pause_time} seconds at minimum position...")
                time.sleep(pause_time)
                
                # Move to maximum Y position
                logger.info(f"Cycle {cycle}: Moving to Y maximum position ({self.y_max}mm)")
                self.move_y_to(self.y_max, speed)
                self.wait_for_movement_complete()
                
                logger.info(f"Cycle {cycle}: Pausing for {pause_time} seconds at maximum position...")
                time.sleep(pause_time)
                
                logger.info(f"Cycle {cycle} completed!")
                
                # Short pause between cycles (except after the last cycle)
                if cycle < cycles:
                    logger.info("Brief pause before next cycle...")
                    time.sleep(2)
            
            # Return to starting position after all cycles
            logger.info(f"\nAll {cycles} cycles completed! Returning to starting position ({start_y:.2f}mm)")
            self.move_y_to(start_y, speed)
            self.wait_for_movement_complete()
            
            logger.info("="*60)
            logger.info(f"Y-AXIS {cycles}-CYCLE TEST COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error during Y-axis {cycles}-cycle test: {e}")
            try:
                logger.info("Attempting to return to starting position...")
                self.move_y_to(start_y)
                self.wait_for_movement_complete()
            except Exception as recovery_error:
                logger.error(f"Failed to return to starting position: {recovery_error}")
            raise


def main():
    """
    Main function to run Y-axis acceleration measurement test with 3 cycles
    """
    # CONFIGURATION - UPDATE THESE VALUES FOR YOUR SETUP
    JUBILEE_IP = "192.168.1.9"  # Your Jubilee's IP address
    TEST_SPEED = 3000  # mm/min - adjust based on your needs
    PAUSE_TIME = 5.0  # seconds to pause at each position
    CYCLES = 3  # Number of back-and-forth cycles
    
    try:
        # Initialize controller
        logger.info("Initializing Jubilee Y-axis controller...")
        jubilee = JubileeYAxisController(JUBILEE_IP)
        
        # Run the test with multiple cycles
        jubilee.run_y_axis_test_3_cycles(pause_time=PAUSE_TIME, speed=TEST_SPEED, cycles=CYCLES)
        
        logger.info(f"Y-axis {CYCLES}-cycle test completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Y-axis test failed: {e}")
    finally:
        logger.info("Y-axis test sequence finished")


if __name__ == "__main__":
    main()