from datetime import datetime, timedelta

import sys

sys.path.append(r'C:/Users/faria/PycharmProjects/simglucose_mod')
import simglucose

from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.user_interface import simulate
from simglucose.controller.bolus_controller import BolusController

ctrller = BolusController(cr=12, cf=15.0360283441)
now = datetime.now()
start_hour = timedelta(hours=float(0))
start_time = datetime.combine(now.date(), datetime.min.time()) + start_hour

sim_time = timedelta(hours=float(24))

simulate(controller=ctrller, patient_names=['adolescent#001'],
         cgm_name="Dexcom",
         cgm_seed=0,
         insulin_pump_name="Insulet",
         start_time=1,
         save_path=None,
         animate='y',
         sim_time=sim_time,
         parallel=False, scenario=RandomScenario(start_time, seed=0))
