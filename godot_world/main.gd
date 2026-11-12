extends Node3D
## FlyDrive's 3D world.
##
## Python owns the simulation (car physics, obstacles, collisions, the fly brain).
## This scene only draws it and sends back what the car's camera sees, so the car
## drives on pixels alone. Protocol, both directions: a 4-byte little-endian length
## followed by the payload (JSON in, raw RGB pixels out).

const HOST := "127.0.0.1"
const DEFAULT_PORT := 47111
const RETRY_SECONDS := 2.0

var tcp := StreamPeerTCP.new()
var port := DEFAULT_PORT
var busy := false
var need := -1
var configured := false
var retry_in := 0.0

var img_size := Vector2i(320, 96)
var cam_forward := 1.9
var cam_height := 1.2
var skin_color := Color(0.85, 0.66, 0.52)

var eye_viewport: SubViewport
var eye_camera: Camera3D
var chase_camera: Camera3D
var car: Node3D
var hud: Label
var eye_view: TextureRect
var obstacle_nodes := {}


func _ready() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--port="):
			port = int(arg.substr(7))
	_build_sky_and_light()
	_build_car()
	_build_cameras()
	_build_hud()
	_connect()


func _connect() -> void:
	tcp = StreamPeerTCP.new()
	tcp.big_endian = false
	var err := tcp.connect_to_host(HOST, port)
	if err != OK:
		push_warning("Could not reach the FlyDrive Python process on port %d" % port)


func _process(delta: float) -> void:
	tcp.poll()
	var status := tcp.get_status()
	if status == StreamPeerTCP.STATUS_CONNECTED:
		_update_chase()
		if not busy:
			_pump()
	elif status == StreamPeerTCP.STATUS_ERROR or status == StreamPeerTCP.STATUS_NONE:
		retry_in -= delta
		if retry_in <= 0.0:
			retry_in = RETRY_SECONDS
			hud.text = ("Waiting for FlyDrive on port %d...\n" % port
					+ "Run:  python run.py --eyes camera --renderer godot --no-launch")
			_connect()


func _pump() -> void:
	if need < 0:
		if tcp.get_available_bytes() < 4:
			return
		need = int(tcp.get_u32())
	if need < 0 or tcp.get_available_bytes() < need:
		return
	var res: Array = tcp.get_data(need)
	need = -1
	if res[0] != OK:
		return
	var bytes: PackedByteArray = res[1]
	var msg = JSON.parse_string(bytes.get_string_from_utf8())
	if typeof(msg) != TYPE_DICTIONARY:
		return
	var kind: String = str(msg.get("type", ""))
	if kind == "hello":
		_configure(msg)
		tcp.put_u32(0)
	elif kind == "frame":
		await _send_frame(msg)
	elif kind == "quit":
		get_tree().quit()


func _send_frame(msg: Dictionary) -> void:
	busy = true
	_apply_state(msg)
	await RenderingServer.frame_post_draw
	var img := eye_viewport.get_texture().get_image()
	img.convert(Image.FORMAT_RGB8)
	var data := img.get_data()
	tcp.put_u32(data.size())
	tcp.put_data(data)
	busy = false


# ---------------------------------------------------------------- world building
func _to3(x: float, y: float, height: float) -> Vector3:
	# Python's world is flat (x east, y north); Godot's Y is up.
	return Vector3(x, height, -y)


func _build_sky_and_light() -> void:
	var env := Environment.new()
	var sky_mat := ProceduralSkyMaterial.new()
	sky_mat.sky_top_color = Color(0.42, 0.62, 0.92)
	sky_mat.sky_horizon_color = Color(0.80, 0.87, 0.95)
	sky_mat.ground_bottom_color = Color(0.30, 0.52, 0.22)
	sky_mat.ground_horizon_color = Color(0.72, 0.80, 0.86)
	var sky := Sky.new()
	sky.sky_material = sky_mat
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.ambient_light_energy = 1.0
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-52, -38, 0)
	sun.light_energy = 1.0
	sun.shadow_enabled = true
	add_child(sun)


func _flat_material(col: Color) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = col
	mat.roughness = 1.0
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	return mat


func _build_ground(colors: Dictionary) -> void:
	var ground := MeshInstance3D.new()
	var plane := PlaneMesh.new()
	plane.size = Vector2(3000, 3000)
	ground.mesh = plane
	ground.position = Vector3(110, 0, 0)
	ground.material_override = _flat_material(_col(colors, "grass"))
	add_child(ground)


func _col(colors: Dictionary, key: String) -> Color:
	var c: Array = colors[key]
	return Color(float(c[0]), float(c[1]), float(c[2]))


func _ribbon(track: Array, inner: float, outer: float, height: float, dashed: bool,
		period: int, on: int) -> ArrayMesh:
	## A flat strip along the track, between `inner` and `outer` metres from the centre.
	## Used for the road itself, the dashed centre line and the solid edge lines.
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := track.size()
	for i in n:
		if dashed and (i % period) >= on:
			continue
		var p: Array = track[i]
		var q: Array = track[(i + 1) % n]
		var px := float(p[0])
		var py := float(p[1])
		var qx := float(q[0])
		var qy := float(q[1])
		var dir := Vector2(qx - px, qy - py).normalized()
		var nrm := Vector2(-dir.y, dir.x)
		for side in [1.0, -1.0]:
			var a := _to3(px + nrm.x * outer * side, py + nrm.y * outer * side, height)
			var b := _to3(px + nrm.x * inner * side, py + nrm.y * inner * side, height)
			var c := _to3(qx + nrm.x * outer * side, qy + nrm.y * outer * side, height)
			var d := _to3(qx + nrm.x * inner * side, qy + nrm.y * inner * side, height)
			for v in [a, c, b, b, c, d]:
				st.set_normal(Vector3.UP)
				st.add_vertex(v)
	return st.commit()


func _add_mesh(mesh: ArrayMesh, col: Color) -> void:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = _flat_material(col)
	add_child(mi)


func _build_road(msg: Dictionary, colors: Dictionary) -> void:
	var track: Array = msg["track"]
	var half := float(msg["road_half"])
	var line_half := float(msg["center_line_half"])
	var dash: Array = msg["dash"]
	var edge: Array = msg["edge_line"]
	_add_mesh(_ribbon(track, 0.0, half, 0.01, false, 1, 1), _col(colors, "road"))
	_add_mesh(_ribbon(track, 0.0, line_half, 0.02, true, int(dash[0]), int(dash[1])),
			_col(colors, "line"))
	_add_mesh(_ribbon(track, float(edge[0]), float(edge[1]), 0.02, false, 1, 1),
			_col(colors, "line"))


func _build_car() -> void:
	car = Node3D.new()
	add_child(car)
	var body := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = Vector3(2.0, 1.1, 4.4)
	body.mesh = box
	body.position.y = 0.55
	body.material_override = _flat_material(Color(0.93, 0.93, 0.95))
	body.layers = 2                      # the car's own camera does not see the car
	car.add_child(body)
	var cabin := MeshInstance3D.new()
	var cbox := BoxMesh.new()
	cbox.size = Vector3(1.7, 0.6, 1.9)
	cabin.mesh = cbox
	cabin.position = Vector3(0, 1.2, 0.2)
	cabin.material_override = _flat_material(Color(0.2, 0.3, 0.45))
	cabin.layers = 2
	car.add_child(cabin)


func _build_cameras() -> void:
	eye_viewport = SubViewport.new()
	eye_viewport.size = img_size
	eye_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(eye_viewport)
	eye_camera = Camera3D.new()
	eye_camera.keep_aspect = Camera3D.KEEP_WIDTH
	eye_camera.fov = 110.0
	eye_camera.near = 0.05
	eye_camera.far = 500.0
	eye_camera.cull_mask = 1
	eye_camera.current = true
	eye_viewport.add_child(eye_camera)

	chase_camera = Camera3D.new()
	chase_camera.fov = 60.0
	chase_camera.far = 1200.0
	chase_camera.current = true
	add_child(chase_camera)


func _build_hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	hud = Label.new()
	hud.position = Vector2(16, 12)
	hud.add_theme_font_size_override("font_size", 18)
	hud.add_theme_color_override("font_color", Color(1, 1, 1))
	hud.add_theme_color_override("font_outline_color", Color(0, 0, 0))
	hud.add_theme_constant_override("outline_size", 6)
	hud.text = "Connecting to FlyDrive..."
	layer.add_child(hud)
	eye_view = TextureRect.new()
	eye_view.position = Vector2(16, 70)
	eye_view.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	eye_view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	eye_view.custom_minimum_size = Vector2(img_size.x * 2, img_size.y * 2)
	eye_view.size = eye_view.custom_minimum_size
	layer.add_child(eye_view)


func _configure(msg: Dictionary) -> void:
	var size: Array = msg["image"]
	img_size = Vector2i(int(size[0]), int(size[1]))
	eye_viewport.size = img_size
	eye_camera.fov = float(msg["hfov"])
	var campos: Array = msg["cam"]
	cam_forward = float(campos[0])
	cam_height = float(campos[1])
	var colors: Dictionary = msg["colors"]
	skin_color = _col(colors, "skin")
	if not configured:
		_build_ground(colors)
		_build_road(msg, colors)
		eye_view.texture = eye_viewport.get_texture()
		eye_view.custom_minimum_size = Vector2(img_size.x * 2, img_size.y * 2)
		eye_view.size = eye_view.custom_minimum_size
	configured = true


# ---------------------------------------------------------------- per frame
func _obstacle_node(kind: int, size: float, height: float, col: Color) -> Node3D:
	var root := Node3D.new()
	if kind == 1:                                     # a person
		var body := MeshInstance3D.new()
		var cyl := CylinderMesh.new()
		cyl.top_radius = size
		cyl.bottom_radius = size
		cyl.height = height - 0.25
		body.mesh = cyl
		body.position.y = (height - 0.25) * 0.5
		body.material_override = _flat_material(col)
		root.add_child(body)
		var head := MeshInstance3D.new()
		var hc := CylinderMesh.new()
		hc.top_radius = size * 0.8
		hc.bottom_radius = size * 0.8
		hc.height = 0.25
		head.mesh = hc
		head.position.y = height - 0.125
		head.material_override = _flat_material(skin_color)
		root.add_child(head)
	else:                                             # a crate
		var box := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = Vector3(size * 2.0, height, size * 2.0)
		box.mesh = bm
		box.position.y = height * 0.5
		box.material_override = _flat_material(col)
		root.add_child(box)
	add_child(root)
	return root


func _apply_state(msg: Dictionary) -> void:
	var c: Array = msg["car"]
	var cx := float(c[0])
	var cy := float(c[1])
	var heading := float(c[2])
	car.position = _to3(cx, cy, 0.0)
	car.rotation.y = heading - PI / 2.0
	eye_camera.position = _to3(cx + cos(heading) * cam_forward, cy + sin(heading) * cam_forward,
			cam_height)
	eye_camera.rotation = Vector3(0.0, heading - PI / 2.0, 0.0)

	var seen := {}
	for row in msg["obstacles"]:
		var uid := int(row[0])
		seen[uid] = true
		var node: Node3D = obstacle_nodes.get(uid)
		if node == null:
			node = _obstacle_node(int(row[1]), float(row[4]), float(row[5]),
					Color(float(row[6]), float(row[7]), float(row[8])))
			obstacle_nodes[uid] = node
		node.position = _to3(float(row[2]), float(row[3]), 0.0)
	for uid in obstacle_nodes.keys():
		if not seen.has(uid):
			obstacle_nodes[uid].queue_free()
			obstacle_nodes.erase(uid)
	hud.text = str(msg.get("hud", ""))


func _update_chase() -> void:
	if car == null:
		return
	var fwd := -car.global_transform.basis.z
	chase_camera.position = car.position - fwd * 11.0 + Vector3.UP * 4.5
	chase_camera.look_at(car.position + fwd * 8.0 + Vector3.UP * 0.8, Vector3.UP)
