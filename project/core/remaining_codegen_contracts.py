"""Compact English action-local API contracts; not complete policy templates.
Each entry describes a real interface side effect. Generated source still selects
and orders calls and is physically validated; no missing calls are auto-inserted.
"""
CONTRACTS={
"return_source":("The arm already holds the upright source cup. Return it to its support, release it, withdraw the hand; do not return HOME.",{
"safe_pose":"safe_pose() -> xyz: read safe-lane waypoint for held object.","move_safe":"move_safe(xyz): carry held object into safe lane. Required before returning across workspace; does not lower or release.",
"target_pose":"target_pose() -> xyz: read final destination surface.","move_above":"move_above(xyz): carry held object above destination, does not lower.",
"align_target":"align_target(xyz): physically refine held-object alignment at destination after move_above and before lower; skipping this fine alignment can leave the returned object outside the goal tolerance.","lower":"lower(xyz): lower held object to destination; argument must be full destination tuple, NOT a gap scalar. Does not release.",
"release":"release(): open gripper and deactivate grasp weld, no arm motion.","retreat":"retreat(): move OPEN hand away; requires release first.","settle":"settle(): wait without grip change."}),
"donor_pick":("Pick up the box with donor arm, finish attached and lifted in the air; receiver remains empty.",{
"grasp_pose":"grasp_pose() -> xyz: read bound donor grasp point.","approach":"approach(xyz): move OPEN hand above grasp.","descend":"descend(xyz): move OPEN hand down to grasp, requires approach.",
"close_gripper":"close_gripper(): close CURRENT hand; requires descent; does not activate weld.","attach_contact_gated":"attach_contact_gated(): validate actual bilateral contact and activate declared weld; requires closed hand.","lift":"lift(): lift ATTACHED object, requires attach_contact_gated."}),
"receive":("Donor already holds the box at meeting pose. Receiver must grasp its own box-face zone and establish dual holding. Do not release donor.",{
"grasp_pose":"grasp_pose() -> xyz: read receiver grasp point.","approach":"approach(xyz): move OPEN receiver above point.","descend":"descend(xyz): lower OPEN receiver to same point.","close_gripper":"close_gripper(): close after descent; does not attach.","attach_contact_gated":"attach_contact_gated(): verify contact and activate receiver weld, creating dual holding."}),
"present":("Donor holds box. Transport to meeting pose and hold, do not release.",{"target_pose":"target_pose() -> xyz: meeting object origin.","move_held":"move_held(xyz): move held object.","settle":"settle(): hold still."}),
"donor_release":("Receiver attachment verified; transfer ownership by releasing donor and withdrawing donor arm. Receiver must keep holding.",{"open_gripper":"open_gripper(): opens DONOR without detaching weld.","detach":"detach(): release DONOR weld after opening.","withdraw":"withdraw(): withdraw DONOR to safe pose after detach."}),
"receiver_place":("Receiver holds box and donor is home. Place box at destination, release and withdraw the receiver safely; do not go HOME.",{
"target_pose":"target_pose() -> xyz: read destination object origin.","carry":"carry(xyz): transport held box above destination.","lower":"lower(xyz): lower held box to support; requires carry; does not release.",
"open_gripper":"open_gripper(): open receiver at CURRENT location after lowering; does not detach weld.","detach":"detach(): remove receiver weld after opening and allow box to settle.","retreat_vertical":"retreat_vertical(): raise empty OPEN hand; requires detach.","withdraw_safe":"withdraw_safe(): move OPEN hand to outer safe pose; requires vertical retreat.","read":"read() -> dict: read-only evaluation; not necessary to execute action."}),
"park":("Arm is EMPTY after releasing its payload. First establish vertical clearance from box walls, then move to parking region. No grasp.",{"target_pose":"target_pose() -> xyz: read parking point.","raise_clear":"raise_clear(): move empty hand vertically up to obstacle clearance; does not park.","move_to":"move_to(xyz): park empty hand, requires explicit raise_clear in same action.","settle":"settle(): hold current commands."}),
"place":("The arm ALREADY HOLDS the item. Put it in its tray slot, open gripper, then vertically withdraw. Do not pick another item or park laterally.",{
"target_pose":"target_pose() -> xyz: destination slot SUPPORT SURFACE point, not grasp point.","move_above":"move_above(xyz): transport CLOSED hand above slot; does not lower.","align_target":"align_target(xyz): refine horizontal alignment above slot.","lower":"lower(xyz): lower held item to slot support; does not release.","place_release":"place_release(): OPEN gripper and remove weld at current location, no motion.","retreat":"retreat(): move EMPTY OPEN hand up, requires prior release.","settle":"settle(): wait without changing gripper."})}
CONTRACTS["return_basin"]=("Arm already holds basin containing caught sphere. Return basin slowly, release and withdraw; do not lose sphere or go HOME.",CONTRACTS["return_source"][1])

def compact(kind,api_docs,description):
    if kind in CONTRACTS:return CONTRACTS[kind]
    return description,api_docs
