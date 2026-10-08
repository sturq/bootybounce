extends Node2D
# Desktop pet: a borderless, transparent window that plays a person loop on top of the taskbar.
# Pets live in pets/<Person>/<Outfit>/ (next to the exe, or in the project when run from the editor).

const SIZES := {"Small": 240, "Medium": 360, "Large": 520}
const GRAVITY := 3000.0
const SETTINGS := "user://settings.cfg"

var pets_dir := ""
var pet_name := ""
var outfit := ""
var size_name := "Medium"
var frames: Array[ImageTexture] = []
var polys: Array[PackedVector2Array] = []
var order: Array[int] = []
var fps := 30.0
var feet := 0  # empty pixels below the feet, so they stand on the taskbar and not above it
var t := 0.0
var shown := -1
var dragging := false
var grab := Vector2i.ZERO
var fall_v := 0.0
var sprite: Sprite2D
var menu: PopupMenu
var args := {}


func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		var kv := a.trim_prefix("--").split("=", true, 1)
		args[kv[0]] = kv[1] if kv.size() > 1 else ""
	pets_dir = ProjectSettings.globalize_path("res://pets") if OS.has_feature("editor") \
		else OS.get_executable_path().get_base_dir().path_join("pets")
	var names := list_pets(pets_dir)
	if names.is_empty():
		OS.alert("No pets found in " + pets_dir, "bootybounce")
		get_tree().quit(1)
		return
	var cfg := ConfigFile.new()
	cfg.load(SETTINGS)
	pet_name = args.get("pet", cfg.get_value("pet", "name", names[0]))
	if pet_name not in names:
		pet_name = names[0]
	outfit = args.get("outfit", cfg.get_value("pet", "outfit", ""))
	if outfit not in list_outfits(pets_dir.path_join(pet_name)):
		outfit = list_outfits(pets_dir.path_join(pet_name))[0]
	size_name = args.get("size", cfg.get_value("pet", "size", size_name))
	if size_name not in SIZES:
		size_name = "Medium"
	get_window().always_on_top = cfg.get_value("pet", "on_top", true)
	sprite = Sprite2D.new()
	sprite.centered = false
	add_child(sprite)
	menu = PopupMenu.new()
	add_child(menu)
	menu.id_pressed.connect(_on_menu)
	load_pet(pets_dir.path_join(pet_name).path_join(outfit), SIZES[size_name])
	var r := usable_rect()
	get_window().position = Vector2i(clampi(cfg.get_value("pet", "x", r.end.x - get_window().size.x - 40),
		r.position.x, r.end.x - get_window().size.x), floor_y())
	if args.has("shot"):
		shoot.call_deferred()


static func list_pets(dir: String) -> PackedStringArray:
	var out := PackedStringArray()
	for d in DirAccess.get_directories_at(dir):
		if not list_outfits(dir.path_join(d)).is_empty():
			out.append(d)
	return out


static func list_outfits(person_dir: String) -> PackedStringArray:
	var out := PackedStringArray()
	for d in DirAccess.get_directories_at(person_dir):
		if FileAccess.file_exists(person_dir.path_join(d).path_join("pet.cfg")):
			out.append(d)
	return out


func load_pet(dir: String, height: int) -> void:
	var c := ConfigFile.new()
	c.load(dir.path_join("pet.cfg"))
	fps = c.get_value("pet", "fps", 30.0)
	var files := Array(DirAccess.get_files_at(dir.path_join("frames"))).filter(func(f): return f.ends_with(".png"))
	files.sort()
	var hit: Array = JSON.parse_string(FileAccess.get_file_as_string(dir.path_join("hit.json")))
	frames.clear()
	polys.clear()
	for i in files.size():
		var img := Image.load_from_file(dir.path_join("frames").path_join(files[i]))
		var s := float(height) / img.get_height()
		if i == 0:
			feet = int((img.get_height() - img.get_used_rect().end.y) * s)
		img.resize(roundi(img.get_width() * s), height, Image.INTERPOLATE_LANCZOS)
		frames.append(ImageTexture.create_from_image(img))
		var p := PackedVector2Array()
		var flat: Array = hit[i]
		for k in range(0, flat.size(), 2):
			p.append(Vector2(flat[k], flat[k + 1]) * s)
		polys.append(p)
	order.assign(range(frames.size()))
	if c.get_value("pet", "pingpong", false):
		order.append_array(range(frames.size() - 2, 0, -1))
	shown = -1
	if is_inside_tree():
		get_window().size = frames[0].get_size()


func usable_rect() -> Rect2i:
	return DisplayServer.screen_get_usable_rect(get_window().current_screen)


func floor_y() -> int:
	return usable_rect().end.y - get_window().size.y + feet


func frame_at(time: float) -> int:
	return order[int(time * fps) % order.size()]


func _process(delta: float) -> void:
	t += delta
	var i := frame_at(t)
	if i != shown:
		shown = i
		sprite.texture = frames[i]
		DisplayServer.window_set_mouse_passthrough(polys[i])
	var w := get_window()
	var pos := w.position
	if dragging:
		pos = DisplayServer.mouse_get_position() - grab
	elif pos.y < floor_y():
		fall_v += GRAVITY * delta
		pos.y = mini(pos.y + int(fall_v * delta), floor_y())
	else:
		fall_v = 0.0
		pos.y = floor_y()
	var r := usable_rect()
	pos.x = clampi(pos.x, r.position.x - w.size.x / 2, r.end.x - w.size.x / 2)
	if pos != w.position:
		w.position = pos


func _input(e: InputEvent) -> void:
	var b := e as InputEventMouseButton
	if b == null:
		return
	if b.button_index == MOUSE_BUTTON_LEFT:
		dragging = b.pressed
		grab = DisplayServer.mouse_get_position() - get_window().position
		if not b.pressed:
			save()
	elif b.button_index == MOUSE_BUTTON_RIGHT and b.pressed:
		build_menu()
		menu.popup(Rect2i(DisplayServer.mouse_get_position(), Vector2i.ZERO))


func build_menu() -> void:
	menu.clear()
	var names := list_pets(pets_dir)
	for k in names.size():
		menu.add_radio_check_item(names[k], k)
		menu.set_item_checked(menu.item_count - 1, names[k] == pet_name)
	menu.add_separator()
	var outfits := list_outfits(pets_dir.path_join(pet_name))
	for k in outfits.size():
		menu.add_radio_check_item(outfits[k], 400 + k)
		menu.set_item_checked(menu.item_count - 1, outfits[k] == outfit)
	menu.add_separator()
	var id := 100
	for s in SIZES:
		menu.add_radio_check_item(s, id)
		menu.set_item_checked(menu.item_count - 1, s == size_name)
		id += 1
	menu.add_separator()
	menu.add_check_item("Always on top", 200)
	menu.set_item_checked(menu.item_count - 1, get_window().always_on_top)
	menu.add_item("Quit", 300)


func _on_menu(id: int) -> void:
	if id >= 400:
		outfit = list_outfits(pets_dir.path_join(pet_name))[id - 400]
	elif id < 100:
		pet_name = list_pets(pets_dir)[id]
		if outfit not in list_outfits(pets_dir.path_join(pet_name)):
			outfit = list_outfits(pets_dir.path_join(pet_name))[0]
	elif id < 200:
		size_name = SIZES.keys()[id - 100]
	elif id == 200:
		get_window().always_on_top = not get_window().always_on_top
	else:
		save()
		get_tree().quit()
		return
	if id < 200 or id >= 400:
		var w := get_window()
		var bottom := w.position.y + w.size.y
		load_pet(pets_dir.path_join(pet_name).path_join(outfit), SIZES[size_name])
		w.position = Vector2i(w.position.x, bottom - w.size.y)
	save()


func save() -> void:
	var cfg := ConfigFile.new()
	cfg.set_value("pet", "name", pet_name)
	cfg.set_value("pet", "outfit", outfit)
	cfg.set_value("pet", "size", size_name)
	cfg.set_value("pet", "on_top", get_window().always_on_top)
	cfg.set_value("pet", "x", get_window().position.x)
	cfg.save(SETTINGS)


# Agent hook: --shot=PATH [--frame=N] [--pet=X] [--outfit=Y] [--size=Z] saves what the window shows (with alpha) and quits.
func shoot() -> void:
	set_process(false)
	var i := int(args.get("frame", "0")) % frames.size()
	sprite.texture = frames[i]
	await RenderingServer.frame_post_draw
	await RenderingServer.frame_post_draw
	get_viewport().get_texture().get_image().save_png(args["shot"])
	print(JSON.stringify({"pet": pet_name, "outfit": outfit, "size": size_name, "frame": i, "frames": frames.size(),
		"loop": order.size(), "window": [get_window().size.x, get_window().size.y], "feet": feet,
		"poly_points": polys[i].size()}))
	get_tree().quit()
