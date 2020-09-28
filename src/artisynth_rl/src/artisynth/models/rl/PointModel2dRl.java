package artisynth.models.rl;

import java.awt.Color;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import artisynth.core.driver.Main;
import artisynth.core.gui.ControlPanel;
import artisynth.core.inverse.ForceTarget;
import artisynth.core.inverse.ForceTargetTerm;
import artisynth.core.inverse.Log;
import artisynth.core.inverse.NetworkHandler;
import artisynth.core.inverse.TrackingController;
import artisynth.core.materials.LinearAxialMuscle;
import artisynth.core.mechmodels.AxialSpring;
import artisynth.core.mechmodels.FrameMarker;
import artisynth.core.mechmodels.MechModel;
import artisynth.core.mechmodels.MechSystemSolver.Integrator;
import artisynth.core.mechmodels.MotionTargetComponent;
import artisynth.core.mechmodels.Muscle;
import artisynth.core.mechmodels.Particle;
import artisynth.core.mechmodels.Point;
import artisynth.core.mechmodels.RigidBody;
import artisynth.core.modelbase.ComponentList;
import artisynth.core.modelbase.Controller;
import artisynth.core.modelbase.Model;
import artisynth.core.modelbase.StepAdjustment;
import artisynth.core.probes.NumericInputProbe;
import artisynth.core.probes.Probe;
import artisynth.core.util.ArtisynthIO;
import artisynth.core.util.ArtisynthPath;
import artisynth.core.workspace.DriverInterface;
import artisynth.core.workspace.PullController;
import artisynth.core.workspace.RootModel;
import artisynth.demos.inverse.PointModel;
import artisynth.models.rl.PointModel2dRl.DemoType;
import maspack.interpolation.CubicSpline;
import maspack.interpolation.Interpolation;
import maspack.interpolation.Interpolation.Order;
import maspack.matrix.Point3d;
import maspack.matrix.RigidTransform3d;
import maspack.matrix.SparseBlockMatrix;
import maspack.matrix.Vector3d;
import maspack.render.RenderProps;
import maspack.render.Renderable;
import maspack.render.Renderer;
import maspack.render.Renderer.LineStyle;
import maspack.render.Renderer.Shading;
import maspack.spatialmotion.SpatialInertia;
import java.lang.Math;
import maspack.interpolation.Interpolation.Order;

public class PointModel2dRl extends RootModel {
	// NumericInputProbe inputProbe; don't need a probe. just set the target
	// position
	static double point_generate_radius = 4.11; // ultimate position with
												// current settings of max
												// muscle excitations
	public static final Vector3d zero = new Vector3d();
	Vector3d disturbance = new Vector3d();

	NetworkHandler networkHandler;
	boolean applyDisturbance = false;
	
	static int size = 2;

	int numMuscles = 12;
	int num_particles = 12;
	
	public enum DemoType {
		Point1d, 
		Point2d, 
		Point3d, 
		NonSym,
	}
	
	static double pointGenerateRadius = 4.11;


	public boolean useReactionForceTargetP = false;

	public static DemoType defaultDemoType = DemoType.Point1d;
	public DemoType myDemoType;

	protected MechModel mech;
	protected FrameMarker center, center_ref;

	String[] muscleLabels = new String[] { "n", "nne", "ne", "ene", "e", "ese",
			"se", "sse", "s", "ssw", "sw", "wsw", "w", "wnw", "nw", "nnw"

	};
	
	String[] muscleLabels1d = new String[] { "M0", "M1"};

	double mass = 0.001; // kg
	double len = 10.0;
	double springK = 10.0;
	double springD = 0.1;
	double springRestLen = len * 0.5;

	double muscleF = 1.0;
	double passiveFraction = 0.1;// 1e-9;
	double muscleOptLen = len * 0.5; // len*1.5);
	double muscleMaxLen = len * 2; // muscleLabels.length;//len*2;
	double muscleD = 0.001;
	double muscleScaleFactor = 1000;
	double pointDamping = 0.1;
	
	double excitationsTime = 0.0;

	boolean excitationsDone = false;
	
	boolean excitationsTimeDone = false;
	
	public PointModel2dRl() {
		Log.DEBUG = true;
	}

	int port = 6020;

	public void build(String[] args) throws IOException {
		build(DemoType.Point1d, args);
	}

	public void build(DemoType demoType, String[] args) {
		myDemoType = demoType;
		System.out.println("myDemoType: " + myDemoType);
		mech = new MechModel("mech");
		mech.setGravity(0, 0, 0);
		mech.setIntegrator(Integrator.Trapezoidal); // todo: why?
		mech.setMaxStepSize(0.01);
		
		createModel(myDemoType);

		setupRenderProps();
		
		createInputProbes();
		
//		setExcitationsWithProbes();
		
		if (myDemoType == DemoType.Point1d) {
			numMuscles = 2;
		}

		for (int i = 0; i < args.length; i += 2)
			if (args[i].equals("-port") == true) {
				port = Integer.parseInt(args[i + 1]);
			}
		Log.info("port: " + port);

		networkHandler = new NetworkHandler(port);
		networkHandler.start();
	}

	public void printType() {
		System.out.println("myType = " + myDemoType.toString());
	}
	
	public String[] generateMuscleLabels() {
		String[] muscleLabels = new String[numMuscles];
		for (int m = 0; m < numMuscles; ++m)
			muscleLabels[m] = "m" + Integer.toString(m);
		return muscleLabels;
	}

	public void createModel(DemoType demoType) {
		String[] muscleLabels = generateMuscleLabels();
		Log.info("muscleLabels length: " + muscleLabels.length);
		switch (demoType) {
		case Point1d: {
			addCenter();
			addCenterRef();
			add1dMuscles(muscleLabels);
			break;
		}
		case Point2d: {
			addCenter();
			addCenterRef();
			add2dLabeledMuscles(muscleLabels);
			break;
		}
		case Point3d: {
			addCenter();
			addCenterRef();
			add3dMuscles();
			break;
		}
		case NonSym: {
			addCenter();
			addCenterRef();
			add2dLabeledMusclesNonSym(muscleLabels);
			break;
		}
		default: {
			System.err.println(
					"PointModel, unknown demo type: " + myDemoType.toString());
		}
		}
		addModel(mech);
	}

	public void setupRenderProps() {
		// set render properties for model

		RenderProps rp = new RenderProps();
		rp.setShading(Shading.SMOOTH);
		rp.setPointStyle(Renderer.PointStyle.SPHERE);
		rp.setPointColor(Color.LIGHT_GRAY);
		rp.setPointRadius(len / 30);
		rp.setLineColor(Color.BLUE.darker());
		// rp.setLineStyle(LineStyle.SPINDLE);
		// rp.setLineRadius(len/25);
		rp.setLineStyle(LineStyle.CYLINDER);
		rp.setLineRadius(0.1604);
		mech.setRenderProps(rp);

		RenderProps.setPointColor(center, Color.BLUE);
		RenderProps.setPointRadius(center, len / 25);
	}

	@Override
	public void addController(Controller controller, Model model) {
		super.addController(controller, model);

		if (controller instanceof PullController) {
			PullController pc = ((PullController) controller);
			pc.setStiffness(20);
			RenderProps.setLineColor(pc, Color.RED.darker());
			RenderProps.setPointColor(pc, Color.RED.darker());
			RenderProps.setLineStyle(pc, LineStyle.SOLID_ARROW);
			RenderProps.setLineRadius(pc, 0.25);
		}
		// mech.addForceEffector ((PullController)controller);
	}

	public void addCenter() {
		RigidBody body = new RigidBody("body_follower");
		body.setInertia(SpatialInertia.createSphereInertia(mass, len / 25));
		mech.addRigidBody(body);
		RenderProps.setVisible(body, true);

		center = new FrameMarker();
		center.setName("center");
		center.setPointDamping(pointDamping);

		RenderProps props = new RenderProps();
		props.setPointColor(Color.BLUE);
		props.setFaceColor(Color.BLUE);
		props.setEdgeColor(Color.BLUE);

		center.setRenderProps(props);

		mech.addFrameMarker(center, body, Point3d.ZERO);

	}

	public void addCenterRef() {
		RigidBody body = new RigidBody("body_ref");
		// body.setInertia (SpatialInertia.createSphereInertia (mass, len/25));
		body.setMass(0);
		mech.addRigidBody(body);
		RenderProps.setVisible(body, true);

		// body.setPosition (new Point3d (5,5,5));

		center_ref = new FrameMarker();
		center_ref.setName("center_ref");
		center_ref.setPointDamping(pointDamping);

		RenderProps props = new RenderProps();
		props.setPointColor(Color.GREEN);
		center_ref.setRenderProps(props);

		mech.addFrameMarker(center_ref, body, Point3d.ZERO);
		// center_ref.setPosition (5,5,5);
	}

	public Point3d getRandomTarget(Point3d center, double radius) {
		Random rand = new Random();

		double theta = rand.nextDouble() * 3.1415;
		double phi = rand.nextDouble() * 3.1415;
		double r = rand.nextDouble() * radius;

		double x = r * Math.cos(theta) * Math.sin(phi);
		double y = r * Math.sin(theta) * Math.sin(phi);
		double z = r * Math.cos(phi);

		// Vector3d targetVec = new Vector3d(rand.nextDouble ()-0.5,
		// rand.nextDouble ()-0.5,
		// rand.nextDouble ()-0.5);
		// targetVec.scale (radius*2);

		Vector3d targetVec = new Vector3d(x, y, z);

		if (myDemoType == DemoType.Point2d || myDemoType == DemoType.NonSym)
			targetVec.y = 0;
		if (myDemoType == DemoType.Point1d)
			targetVec.z = 0;

		Point3d targetPnt = new Point3d(targetVec.x, targetVec.y, targetVec.z);
		return targetPnt;
	}

	public StepAdjustment advance(double t0, double t1, int flags) {
		JSONObject jo_receive = networkHandler.getMessage();
		if (jo_receive != null) {
			try {
				Log.info("advance: jo_receive = " + jo_receive.getString("type"));
				switch (jo_receive.getString("type")) {
				case "getActionSize":
					sendActionSize();
					break;
				case "reset":
					try {
						resetRefPosition();
					} catch (Exception e) {
						e.printStackTrace();
					}
					break;
				case "setExcitations":
					setExcitations(jo_receive);
					setExcitationsWithProbes();
//					boolean set_excitations_done = setExcitationsWithProbes(jo_receive);
//					if (set_excitations_done) {
//						sendExcitationsDone();
//					}
					break;
				case "setExcitationsTime":
					boolean set_time_done = setExcitationsTime(jo_receive);
					if (set_time_done) {
						sendExcitationsTimeDone();
					}
					break;
				case "getState":
					sendState();
					break;
				case "setName":
					String name = jo_receive.getString("name");
					mech.setName(name);
				default:
					break;
				}
			} catch (JSONException e) {
				Log.info("Error in advance: " + e.getMessage());
			}
		}
		return super.advance(t0, t1, flags);
	}
	

	private double getTime() {
		return Main.getMain().getTime();
	}
	
	private double getDistanceError(double[] target, double[] source) {
		double sum = 0.0;
		double distance = 0.0;
		assert target.length == source.length;
		
		for(int i=0;i<target.length;i++)
		{
			sum = sum + Math.pow((source[i]-target[i]),2.0);
			distance = Math.sqrt(sum);
		}
		return distance;
	}
	
	

	private void sendState() {
		// try {
		// Thread.sleep(200);
		// } catch (InterruptedException e) {
		// Log.log("Error in sleep sendState: " + e.getMessage());
		// }
		JSONObject jo_send_state = new JSONObject();
		RigidBody body_ref = mech.rigidBodies().get("body_ref");
		RigidBody body_follower = mech.rigidBodies().get("body_follower");
		try {
			jo_send_state.put("type", "state");

			double[] refPosArr = new double[3];
			body_ref.getPosition().get(refPosArr);
			jo_send_state.put("ref_pos", refPosArr);

			double[] followPosArr = new double[3];
			body_follower.getPosition().get(followPosArr);
			jo_send_state.put("follow_pos", followPosArr);

			double myTime = getTime();
			jo_send_state.put("time", myTime);
			
			double myDistanceError = getDistanceError(refPosArr, followPosArr);
			jo_send_state.put("distanceError", myDistanceError);
			
			networkHandler.send(jo_send_state);
		} catch (JSONException e) {
			System.out.println("Error in send: " + e.getMessage());
		}
	}
	
	private void sendActionSize() {
		JSONObject jo_send_action_size = new JSONObject();
		
		try {
			jo_send_action_size.put("type", "actionSize");
			jo_send_action_size.put("actionSize", numMuscles);
			networkHandler.send(jo_send_action_size);
		} catch (JSONException e) {
			System.out.println("Error in send: " + e.getMessage());
		}
	}

	@Override
	public void finalize() throws Throwable {
		networkHandler.closeConnection();
		super.finalize();

	}

	private void resetRefPosition() throws Exception {
		RigidBody body_ref = mech.rigidBodies().get("body_ref");
//		Point3d pos = getRandomTarget(new Point3d(0, 0, 0),
//				point_generate_radius);
		double[] linear_weights = LinearPosWeightGen();
		
		Point3d lin_pos = getRandomLinearTarget(linear_weights);
//		body_ref.setPosition(pos);
		body_ref.setPosition(lin_pos);
	}
	

	private void setExcitations(JSONObject jo_rec) throws JSONException {
		for (String label : muscleLabels1d) {
			AxialSpring m = mech.axialSprings().get(label);
			{
				Log.info("exication set: " + label);
				if (m instanceof Muscle)
					((Muscle) m).setExcitation(jo_rec
							.getJSONObject("excitations").getDouble(label));
			}
		}
		log("Exications filled");

	}
	
	
	private ComponentList<NumericInputProbe> getProbes(){
		ComponentList<NumericInputProbe> probes = new ComponentList<NumericInputProbe>(NumericInputProbe.class);
		for(Probe p: getInputProbes()) {
			probes.add((NumericInputProbe) p); 
		}
		return probes;
	}
	
	private boolean setExcitationsWithProbes(JSONObject jo_rec) throws JSONException {
		double exsTime = getExcitationsTime();  // get the time value for the application of the muscle activation values
		
		Log.debug("exsTime is: " + exsTime);
		Log.debug("input probes: " + getInputProbes().get(0));
		
		ComponentList<NumericInputProbe> inputProbes = new ComponentList<NumericInputProbe>(NumericInputProbe.class);
		inputProbes = getProbes();
		NumericInputProbe p0 = inputProbes.get(0);
		NumericInputProbe p1 = inputProbes.get(1);
		
		p0.setInterpolationOrder(Order.Cubic);
		p1.setInterpolationOrder(Order.Cubic);
		
		p0.setStopTime(exsTime);
		p1.setStopTime(exsTime);
		
//		p0.addData(0.0, new double[] {0.0});
		p0.addData(exsTime, new double[] {jo_rec.getJSONObject("excitations").getDouble(muscleLabels1d[0])});
//		p1.addData(0.0, new double[] {0.0});
		p1.addData(exsTime, new double[] {jo_rec.getJSONObject("excitations").getDouble(muscleLabels1d[1])});
		
		
		// flag that excitations are done and remove all the input probes
		this.excitationsDone = true;
		
		return this.excitationsDone;
	}
	
	
	public void setExcitationsWithProbes() {
		double exsTime = 1.5;  // get the time value for the application of the muscle activation values
		
		Log.debug("exsTime is: " + exsTime);
		Log.debug("input probes: " + getInputProbes().get(0));
		
		ComponentList<NumericInputProbe> inputProbes = new ComponentList<NumericInputProbe>(NumericInputProbe.class);
		inputProbes = getProbes();
		NumericInputProbe p0 = inputProbes.get(0);
		NumericInputProbe p1 = inputProbes.get(1);
		
		
		p0.setInterpolationOrder(Order.Cubic);
		p1.setInterpolationOrder(Order.Cubic);
		
//		p0.apply(exsTime);
//		p1.apply(exsTime);
		
//		p0.setStopTime(exsTime);
//		p1.setStopTime(exsTime);
		
//		p0.addData(0.0, new double[] {0.0});
		p0.addData(exsTime, new double[] {0.7});
//		p1.addData(0.0, new double[] {0.0});
		p1.addData(exsTime, new double[] {0.1});
		
		
	}
	
	private boolean setExcitationsTime(JSONObject jo_rec) throws JSONException {
		this.excitationsTime = jo_rec.getJSONObject("excitationsTime").getDouble("excitationsTime0");
		Log.info("muscle duration is: " + this.excitationsTime);
		this.excitationsTimeDone = true;
		
		return this.excitationsTimeDone;
	}
	
	private void sendExcitationsTimeDone() throws JSONException {
		JSONObject jo_send_excitations_time_done = new JSONObject();
		jo_send_excitations_time_done.put("type", "sendExcitationsTimeDone");
		jo_send_excitations_time_done.put("sendExcitationsTimeDone", this.excitationsTimeDone);
		networkHandler.send(jo_send_excitations_time_done);	
	}
	
	private void sendExcitationsDone() throws JSONException{
		JSONObject jo_send_excitations_time_done = new JSONObject();
		jo_send_excitations_time_done.put("type", "sendExcitationsDone");
		jo_send_excitations_time_done.put("sendExcitationsDone", this.excitationsDone);
		networkHandler.send(jo_send_excitations_time_done);
	}
	
	
	private double getExcitationsTime() {
		return this.excitationsTime;
	}
	
	
	public class getTargetReachStatus extends Thread{
		/* Separate process/thread that continuously checks state of the world 
		 If the target is acquired interrupts the simulation, generates a new goal and 
		sends out a request to client to send a new motor program a */
		
		boolean done = false;
		
		
	}

	public void log(Object obj) {
		System.out.println(obj);
	}

	public void add2dLabeledMusclesNonSym(String[] labels) {

		// if (applyDisturbance) {
		// } else {
		// }

		addMusclesNonSym(new RigidTransform3d(), labels.length, 0.0);
		int i = 0;
		for (AxialSpring s : mech.axialSprings()) {
			if (s instanceof Muscle) {
				s.setName(labels[i]);
				// ((Muscle) s).setMaxForce(muscleFmult * muscleFs[i]);
				// RenderProps.setLineRadius(s, 0.1 * muscleFs[i]);
				i += 1;
			}
		}
	}

	public void add2dLabeledMuscles(String[] labels) {

		// if (applyDisturbance) {
		// } else {
		// }

		addMuscles(new RigidTransform3d(), labels.length, 0.0);
		int i = 0;
		for (AxialSpring s : mech.axialSprings()) {
			if (s instanceof Muscle) {
				s.setName(labels[i]);
				// ((Muscle) s).setMaxForce(muscleFmult * muscleFs[i]);
				// RenderProps.setLineRadius(s, 0.1 * muscleFs[i]);
				i += 1;
			}
		}
	}

	public void add3dMuscles() {
		int[] x = new int[] { -1, 0, 1 };
		int[] y = new int[] { -1, 0, 1 };
		int[] z = new int[] { -1, 0, 1 };
		double eps = 1e-4;

		int muscleCount = 0;
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < y.length; j++) {
				for (int k = 0; k < z.length; k++) {
					Point3d pnt = new Point3d(x[i], y[j], z[k]);
					if (pnt.x == 0 || pnt.y == 0 || pnt.z == 0)
						continue;
					// if (pnt.norm() < 1e-4 || pnt.norm() > Math.sqrt(2))
					// continue;
					// if (pnt.norm() < 1e-4 || pnt.norm() > 1.0)
					// continue;
					pnt.normalize();
					pnt.scale(len);
					Particle endPt = new Particle(mass, pnt);
					endPt.setDynamic(false);
					mech.addParticle(endPt);
					Muscle m = addMuscle(endPt);
					// m.setName(String.format("x%dy%dz%d",x[i],y[j],z[k]));
					m.setName("m" + Integer.toString(muscleCount++));
					RenderProps.setLineColor(m, Color.RED);
				}
			}
		}

	}

	public void add1dMuscles(String[] labels) {
		
		boolean[] dyn = new boolean[] { false, false };
		int[] x = new int[] { -1, 1 };

		ArrayList<Point> pts = new ArrayList<Point>(x.length);
		for (int i = 0; i < x.length; i++) {
			Point3d pnt = new Point3d(x[i], 0, 0);

			pnt.scale(len);
			Particle pt = new Particle(mass, pnt);
			pt.setPointDamping(pointDamping);
			pt.setDynamic(dyn[i]);
			mech.addParticle(pt);
			pts.add(pt);
			addMuscle(pt);
		}
		int k = 0;
		for (AxialSpring m : mech.axialSprings()) {
			if (m instanceof Muscle) {
				m.setName(labels[k]);
				k += 1;
			}
		}

	}


	public void addMuscles() {
		addMuscles(new RigidTransform3d(), 2, 0.0);
	}

	public void addMusclesNonSym(RigidTransform3d X, int num, double offset) {

//		double[] disturb = new double[num];
//		for (int i = 0; i<num; ++i)
//		{
//			double sign = 1;
//			if (i%2==0) sign = -1;
//			disturb[i] = 0.5 + (i%6+1) * (sign)*((double)i)/4;
//		}

		double[] disturb = { 1.5, -0.5, 2.5, -0.5, -0.4, -0.5, 0.5, 0.5, 2.5,
				-0.2, 0.5 };

		for (int i = 0; i < num; i++) {
			double degree = 2 * Math.PI * ((double) i / num) + disturb[i];

			Point3d pnt = new Point3d((len + disturb[i]) * Math.sin(degree),
					0.0, (len + disturb[i]) * Math.cos(degree));
			pnt.transform(X.R);
			Particle fixed = new Particle(mass, pnt);
			fixed.setDynamic(false);
			mech.addParticle(fixed);
			// System.out.println (pnt);

			addMuscle(fixed);
		}
	}

	public void addMuscles(RigidTransform3d X, int num, double offset) {

		for (int i = 0; i < num; i++) {
			double degree = 2 * Math.PI * ((double) i / num);

			Point3d pnt = new Point3d(len * Math.sin(degree), 0.0,
					len * Math.cos(degree));
			pnt.transform(X.R);
			Particle fixed = new Particle(mass, pnt);
			fixed.setDynamic(false);
			mech.addParticle(fixed);
			// System.out.println (pnt);

			addMuscle(fixed);
		}
	}

	private Muscle addMuscle(Point endPt) {
		return addMuscle(endPt, center);
	}

	private Muscle addMuscle(Point p0, Point p1) {
		// Muscle m = Muscle.createLinear(muscleF, muscleMaxLen);
		Muscle m = new Muscle();
		// ConstantAxialMuscleMaterial mat = new ConstantAxialMuscleMaterial();
		LinearAxialMuscle mat = new LinearAxialMuscle();
		// PeckAxialMuscleMaterial mat = new PeckAxialMuscleMaterial();
		mat.setMaxForce(muscleF);
		mat.setMaxLength(muscleMaxLen);
		mat.setDamping(muscleD);
		mat.setOptLength(muscleOptLen);
		mat.setPassiveFraction(passiveFraction);
		mat.setForceScaling(muscleScaleFactor);
		m.setMaterial(mat);
		m.setRestLength(len);
		m.setFirstPoint(p0);
		m.setSecondPoint(p1);
		mech.addAxialSpring(m);
		RenderProps.setLineColor(m, Color.RED);

		return m;
	}

	public MechModel getMechModel() {
		return mech;
	}

	@Override
	public void attach(DriverInterface driver) {
		super.attach(driver);

		if (getControlPanels().size() == 0) {
			ControlPanel panel = new ControlPanel("activations", "");
			for (AxialSpring s : mech.axialSprings()) {
				if (s instanceof Muscle) {
					Muscle m = (Muscle) s;
					String name = (m.getName() == null ? "m" + m.getNumber()
							: m.getName().toUpperCase());
					panel.addWidget(name, m, "excitation", 0.0, 1.0);
				}
			}
			addControlPanel(panel);
		}
	}

	public void addTrackingController() {
		TrackingController myTrackingController = new TrackingController(mech,
				"tcon");
		for (AxialSpring s : mech.axialSprings()) {
			if (s instanceof Muscle) {
				myTrackingController.addExciter((Muscle) s);
			}
		}

		myTrackingController.addL2RegularizationTerm();
		MotionTargetComponent target = myTrackingController
				.addMotionTarget(center);
		RenderProps.setPointRadius((Renderable) target, 0.525);

		if (useReactionForceTargetP) {
			ForceTargetTerm forceTerm = new ForceTargetTerm(
					myTrackingController);
			ForceTarget ft = forceTerm.addForceTarget(
					mech.bodyConnectors().get("center_constraint"));
			ft.setArrowSize(2);
			RenderProps.setLineStyle(ft, LineStyle.CYLINDER);
			RenderProps.setLineRadius(ft, 0.25);
			forceTerm.setWeight(1d);
			myTrackingController.addForceTargetTerm(forceTerm);
		}

		// myTrackingController.getSolver().setBounds(0.01, 0.99);

		myTrackingController.createProbesAndPanel(this);
		addController(myTrackingController);
	}
	
	public double[] LinearPosWeightGen() throws Exception {
		
		double[] weights = new double[size];
		double[] linear_pos_weights = new double[size];
		double weights_sum = 0.0;
		double sum = 0.0;
		
		
		for (int i = 0; i < weights.length; i++) {
	        weights[i] = Math.random();
	    }
		
		for(double w : weights){
			weights_sum += w;
			}
			
			
		for (int i = 0; i < weights.length; i++) {
			linear_pos_weights[i] = weights[i] / weights_sum;
	    }
		
		
		for(double pw : linear_pos_weights){
			sum += pw;
			}
		
		// test case to check weight sum up to 1

		if(Math.rint(sum) != 1.0) {
			throw new Exception("Sum of weights should be equal to 1.0");
		}
		
		return linear_pos_weights;
		}

   public Point3d getRandomLinearTarget(double[] linear_weights) {
		
	Point3d LinearPos = new Point3d(0.0, 0.0, 0.0);
	Point3d ExtremeLeftPos = new Point3d(-pointGenerateRadius, 0.0, 0.0);
	Point3d ExtremeRightPos = new Point3d(pointGenerateRadius, 0.0, 0.0);


	Vector3d left = ((Vector3d)ExtremeLeftPos).scale(linear_weights[0]);
	Vector3d right = ((Vector3d)ExtremeRightPos).scale(linear_weights[1]);
	Vector3d target = ((Vector3d)ExtremeRightPos);
	
	target.add(left, right);
	
	LinearPos =  new Point3d(target.x, target.y, target.z);
	
	return LinearPos;
	
	}
   
   public void createInputProbes() {
		
		NumericInputProbe m0probe =
				new NumericInputProbe (
						mech, "axialSprings/m0:excitation", 0, 54000);
		NumericInputProbe m1probe =
				new NumericInputProbe (
						mech, "axialSprings/m1:excitation", 0, 54000);

		m0probe.setName("m0 excitation probe");
		m1probe.setName("m1 excitation probe");
		
		addInputProbe(m0probe);
		addInputProbe(m1probe);
		
		Log.debug("Input Probes: " + getInputProbes().get(0).getName());
	}

	public void loadProbes() {
		String probeFileFullPath = ArtisynthPath.getWorkingDir().getPath()
				+ "/0probes.art";
		System.out.println("Loading Probes from File: " + probeFileFullPath);

		try {
			scanProbes(ArtisynthIO.newReaderTokenizer(probeFileFullPath));
		} catch (Exception e) {
			System.out.println("Error reading probe file");
			e.printStackTrace();
		}
	}

}
