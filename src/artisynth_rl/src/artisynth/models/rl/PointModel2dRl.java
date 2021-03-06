package artisynth.models.rl;

import java.awt.Color;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

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
import artisynth.core.modelbase.Controller;
import artisynth.core.modelbase.Model;
import artisynth.core.modelbase.StepAdjustment;
import artisynth.core.util.ArtisynthIO;
import artisynth.core.util.ArtisynthPath;
import artisynth.core.workspace.DriverInterface;
import artisynth.core.workspace.PullController;
import artisynth.core.workspace.RootModel;
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

	public enum DemoType {
		Point1d, Point2d, Point3d, NonSym,
	}

	public boolean useReactionForceTargetP = false;

	public static DemoType defaultDemoType = DemoType.Point2d;
	protected DemoType myDemoType;

	protected MechModel mech;
	protected FrameMarker center, center_ref;

	String[] muscleLabels = new String[] { "n", "nne", "ne", "ene", "e", "ese",
			"se", "sse", "s", "ssw", "sw", "wsw", "w", "wnw", "nw", "nnw"

	};

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

	public PointModel2dRl() {
		Log.logging = true;
	}

	int port = 6020;

	public void build(String[] args) throws IOException {
		build(defaultDemoType, args);
	}

	public void build(DemoType demoType, String[] args) {
		myDemoType = demoType;

		mech = new MechModel("mech");
		mech.setGravity(0, 0, 0);
		mech.setIntegrator(Integrator.Trapezoidal); // todo: why?
		mech.setMaxStepSize(0.01);

		createModel(myDemoType);

		setupRenderProps();

		for (int i = 0; i < args.length; i += 2)
			if (args[i].equals("-port") == true) {
				port = Integer.parseInt(args[i + 1]);
			}

		networkHandler = new NetworkHandler(port);
		networkHandler.start();
	}

	public void printType() {
		System.out.println("myType = " + myDemoType.toString());
	}

	public void createModel(DemoType demoType) {
		switch (demoType) {
		case Point1d: {
			add1dMuscles();
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
				log("advance: jo_receive = " + jo_receive.getString("type"));
				switch (jo_receive.getString("type")) {
				case "reset":
					resetRefPosition();
					break;
				case "setExcitations":
					setExcitations(jo_receive);
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
				log("Error in advance: " + e.getMessage());
			}
		}
		return super.advance(t0, t1, flags);
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

//			jo_send_state.put("follow_pos", body_follower.getPosition());
			// print(jo.toString ());
			networkHandler.send(jo_send_state);
		} catch (JSONException e) {
			System.out.println("Error in send: " + e.getMessage());
		}
	}

	@Override
	public void finalize() throws Throwable {
		networkHandler.closeConnection();
		super.finalize();

	}

	private void resetRefPosition() {
		RigidBody body_ref = mech.rigidBodies().get("body_ref");
		Point3d pos = getRandomTarget(new Point3d(0, 0, 0),
				point_generate_radius);
		body_ref.setPosition(pos);
	}

	private void setExcitations(JSONObject jo_rec) throws JSONException {
		for (String label : muscleLabels) {
			AxialSpring m = mech.axialSprings().get(label);
			{
				log("exication set: " + label);
				if (m instanceof Muscle)
					((Muscle) m).setExcitation(jo_rec
							.getJSONObject("excitations").getDouble(label));
			}
		}
		log("Exications filled");

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

	public void add1dMuscles() {
		boolean[] dyn = new boolean[] { false, true, false };
		int[] x = new int[] { -1, 0, 1 };

		// boolean[] dyn = new boolean[]{false,true, true, true,false};
		// int[] x = new int[]{-2,-1,0,1,2};
		// double eps = 1e-4;

		ArrayList<Point> pts = new ArrayList<Point>(x.length);
		for (int i = 0; i < x.length; i++) {
			Point3d pnt = new Point3d(x[i], 0, 0);

			// pnt.normalize();
			pnt.scale(len);
			Particle pt = new Particle(mass, pnt);
			pt.setPointDamping(pointDamping);
			pt.setDynamic(dyn[i]);
			mech.addParticle(pt);
			pts.add(pt);

			if (x[i] == 0) {
				// center = pt;
			}
		}

		for (int i = 1; i < pts.size(); i++) {
			AxialSpring m;
			Point p0 = pts.get(i - 1);
			Point p1 = pts.get(i);
			// if (p0==center || p1==center)
			// m = addAxialSpring(p0, p1);
			// else
			m = addMuscle(p0, p1);
			m.setName("m" + Integer.toString(m.getNumber()));
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
