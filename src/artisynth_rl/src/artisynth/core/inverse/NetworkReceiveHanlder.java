package artisynth.core.inverse;

import java.io.IOException;
import java.io.InputStream;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.locks.ReentrantLock;

import org.json.JSONException;
import org.json.JSONObject;

public class NetworkReceiveHanlder extends Thread {
	InputStream in;
	Queue<JSONObject> queue;
	public ReentrantLock lock = new ReentrantLock();
	byte[] b = new byte[100000];
	public Boolean exit;

	public NetworkReceiveHanlder(InputStream socket) {
		this.in = socket;
		queue = new LinkedList<JSONObject>();
		this.exit = false;
	}
		
	public void stop_thread() {
		this.exit = true;
	}

	public void run() {
		while (!exit) {
			try {
				JSONObject jo = receiveJsonObject();

				if (jo != null) {
					Log.info("Obj received: " + jo.toString());
					// if (lock.isLocked())
					// return;
					try {
						Log.info("Locking lock in run");
						lock.lock();
						queue.add(jo);
						Log.info("Queue size after add: " + queue.size());
					} catch (Exception e) {
						Log.info("Error in NetowkrReceive run: "
								+ e.getMessage());
					} finally {
						// lock.notify();
						lock.unlock();
						Log.info("Unlocked lock in run");
					}
				}

			} catch (SocketException e) {
				Log.info("SocketException in receiveJsonObject: "
						+ e.getMessage());
				try {
					in.close();
					this.interrupt();
					in = null;
					Log.info("Closing the receive thread");
					break;
				} catch (IOException ioerr) {
					Log.info("Error in closing the receive thread");
				}
			} catch (IOException e) {
				Log.info("IOException in receiveJsonObject: " + e.getMessage());
			} catch (JSONException e) {
				Log.info("JSONException in receiveJsonObject: "
						+ e.getMessage());
			}
		}
	}

	protected JSONObject getMessage() {

		JSONObject jo = null;
		//log("NetworkReceiveHandler.getMessage: queue.size=" + queue.size() + 
		//		" empty?= " + queue.isEmpty());
		if (queue.size() > 0) {
			if (lock.isLocked()) {
				log("NetworkReceiveHandler.getMessage: lock is locked");
				return null;
			}
			try {
				Log.info("Locking lock in getMessage");
				lock.lock();
				jo = queue.remove();
				Log.info("Removed from Queue: " + jo.getString("type"));
				Log.info("Queue size after remove: " + queue.size());
			} catch (Exception e) {
				Log.info("Exception in getMessage: " + e.getMessage());
			} finally {
				// lock.notify();
				lock.unlock();
				Log.info("Unlocked lock in getMessage");
			}
		}
		return jo;
	}

	private JSONObject receiveJsonObject()
			throws JSONException, IOException, SocketException {
		if (in == null)
			throw new SocketException("Socket is closed");
		byte[] int_bytes = new byte[4];
		int bBytesToRead = in.read(int_bytes, 0, 4);
		if (bBytesToRead <= 0)
			return null;
		assert (bBytesToRead == 4);
		ByteBuffer wrapped = ByteBuffer.wrap(int_bytes);

		int bytesToRead = wrapped.getInt();
		//System.out.println("bytesToRead: " + bytesToRead);
		int numbytes = in.read(b, 0, bytesToRead);
		if (numbytes <= 0)
			return null;
		JSONObject jo = null;
		try {
			jo = new JSONObject(new String(b));
		} catch (JSONException e) {
			Log.info("Error in receiveJsonObject: " + e.getMessage());
			throw new JSONException(new String(b));
		}
		return jo;
	}
	public void log(Object obj) {
		//System.out.println(obj);
	}
}
