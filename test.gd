extends SceneTree
# Headless check of every pet: godot --headless --path . --script res://test.gd

const Pet := preload("res://pet.gd")


func _init() -> void:
	var fails := 0
	var dir := ProjectSettings.globalize_path("res://pets")
	var names := Pet.list_pets(dir)
	if names.is_empty():
		fails += 1
		printerr("no pets in ", dir)
	for name in names:
		var p := Pet.new()
		p.load_pet(dir.path_join(name), 360)
		var n := p.frames.size()
		var bad := PackedStringArray()
		if n < 2:
			bad.append("only %d frames" % n)
		if p.polys.size() != n:
			bad.append("%d click polygons for %d frames" % [p.polys.size(), n])
		for i in mini(n, p.polys.size()):
			if p.frames[i].get_height() != 360 or p.frames[i].get_size() != p.frames[0].get_size():
				bad.append("frame %d is %s" % [i, p.frames[i].get_size()])
			if p.polys[i].size() < 3:
				bad.append("frame %d has no click polygon" % i)
		# the loop never jumps: each step moves one frame (a cut loop may wrap from the last to the first)
		for k in p.order.size():
			var step: int = absi(p.order[(k + 1) % p.order.size()] - p.order[k])
			if step != 1 and step != n - 1:
				bad.append("loop jumps %d frames at step %d" % [step, k])
		if p.feet < 0 or p.feet > 40:
			bad.append("feet margin %d px" % p.feet)
		print("%s: %d frames, loop %d, %.2f s, %s" % [name, n, p.order.size(), p.order.size() / p.fps,
			"ok" if bad.is_empty() else "FAIL " + ", ".join(bad)])
		fails += int(not bad.is_empty())
		p.free()
	quit(1 if fails else 0)
