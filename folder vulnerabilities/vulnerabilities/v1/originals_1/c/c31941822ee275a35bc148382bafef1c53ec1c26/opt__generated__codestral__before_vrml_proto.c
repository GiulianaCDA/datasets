 /*This file is part of GPAC, (c) 2001-2008 Thomas Digregorio / 2009-2014 GPAC Team / Telecom ParisTech.*/


#include <gpac/internal/scenegraph_dev.h>
#include <gpac/nodes_vrml.h>
#if !defined(GPAC_DISABLE_VRML) && !defined(GPAC_DISABLE_CORE_TOOLS)

GF_Err gf_sg_proto_init_graph(GF_ProtoInstance *inst, Bool allow_reinit);

static void proto_set_parent(void *node, GF_SceneGraph *sg, NodeID parent)
{
	((GF_Node *)node)->sgprivate->ParentNode = (GF_Node*)parent;
	if (parent && ((GF_Node*)node)->sgprivate->scenegraph->RootNode == node)
		((GF_Node*)node)->sgprivate->scenegraph->RootNode = (void *)parent;
}
static void proto_get_parent(void *node, GF_SceneGraph *sg, NodeID *p_parent)
{
	*p_parent = ((GF_Node *) node)->sgprivate->ParentNode ? ((GF_Node*)node)->sgprivate->ParentNode->sgprivate->NodeID : 0;
}
static void proto_get_id(void *node, GF_SceneGraph *sg, NodeID *nid)
{
	*nid = ((GF_Node *) node)->sgprivate->NodeID;
}
static void proto_get_tag(void *node, u32 *tag)
{
	*tag = ((GF_Node*)node)->sgprivate->tag;
}
static GF_Err proto_add_child(void *_parent, NodeID node)
{
	GF_ProtoInstance *parent = (GF_ProtoInstance *) _parent;
	return gf_list_insert(parent->nodes, parent->nodes ? gf_list_count(parent->nodes) : 0, (GF_Node*)node);
}
static GF_Err proto_add_child_at(void *_parent, NodeID node, u32 index)
{
	GF_ProtoInstance *parent = (GF_ProtoInstance *) _parent;
	return gf_list_insert(parent->nodes, index, (GF_Node*)node);
}
static GF_Err proto_remove_child(void *_parent, NodeID node)
{
	GF_ProtoInstance *parent = (GF_ProtoInstance *) _parent;
	u32 i=0;
	while (((GF_Node*)gf_list_enum(parent->nodes, &i)) != (GF_Node*)node) { }
	return gf_list_removen(parent->nodes, i-1);
}
static u32 proto_get_child_count(void *_parent)
{
	GF_ProtoInstance *parent = (GF_ProtoInstance *) _parent;
	if (!parent->nodes) return 0;
	return gf_list_count(parent->nodes);
}
static GF_Err proto_get_child_index(void *_parent, NodeID node, u32 *index)
{
	GF_ProtoInstance *parent = (GF_ProtoInstance *) _parent;
	u32 i=0;
	while (((GF_Node*)gf_list_enum(parent->nodes, &i)) != (GF_Node*)node) { }
	*index = i-1;
	return GF_OK;
}
static NodeID proto_get_child(void *_parent, u32 index)
{
	GF_ProtoInstance *parent = (GF_ProtoInstance *) _parent;
	if (!parent->nodes) return 0;
	return (NodeID) gf_list_get(parent->nodes, index);
}
static Bool proto_same_graph(void *node1, void *node2)
{
	return ((GF_Node*)node1)->sgprivate->scenegraph == ((GF_Node *) node2)->sgprivate->scenegraph;
}
static GF_Err proto_check_child(void *_parent, NodeID child, Bool allow_root)
{
	if (!allow_root && !((GF_Node*)child)->sgprivate->ParentNode) return GF_BAD_PARAM;
	return GF_OK;
}
static void proto_set_class(void *node, u32 class_type)
{
	((GF_Node *) node)->sgprivate->tag = class_type;
}
static u8 proto_get_in_flags(void *node)
{
	return ((GF_Node*)node)->sgprivate->in_flags;
}
static void proto_set_in_flags(void *node, u8 flags)
{
	((GF_Node*)node)->sgprivate->in_flags = flags;
}
static Bool proto_get_name_count(void *n, u32 *nb_names)
{
	*nb_names = gf_list_count(((GF_Node *) n)->sgprivate->NameStack);
	return 1;
}
static const char *proto_get_name(void *n, u32 name_index)
{
	return (const char*)gf_list_get(((GF_Node *) n)->sgprivate->NameStack, name_index);
}
static void proto_set_name(void *n, const char *name)
{
	if (((GF_Node *) n)->sgprivate->NameStack) gf_free(((GF_Node *) n)->sgprivate->NameStack);
	((GF_Node *) n)->sgprivate->NameStack = NULL;
	gf_list_add(((GF_Node *) n)->sgprivate->NameStack, name);
}
static void proto_remove_name(void *n)
{
	if (((GF_Node *) n)->sgprivate->NameStack) gf_free(((GF_Node *) n)->sgprivate->NameStack);
	((GF_Node *) n)->sgprivate->NameStack = NULL;
}
static void proto_add_name(void *n, const char *name)
{
	gf_list_add(((GF_Node *) n)->sgprivate->NameStack, name);
}
static Bool proto_check_node_flags(void *node, u16 flags)
{
	return ((GF_Node *) node)->sgprivate->flags & flags;
}
static void proto_set_node_flags(void *node, u16 flags)
{
	((GF_Node *) node)->sgprivate->flags |= flags;
}
static void proto_reset_node_flags(void *node, u16 flags)
{
	((GF_Node *) node)->sgprivate->flags &= ~flags;
}

static GF_ProtoInterface proto_ifce = {
	proto_set_parent,
	proto_get_parent,
	proto_get_id,
	proto_add_child,
	proto_remove_child,
	proto_get_tag,
	proto_same_graph,
	proto_check_child,
	proto_get_child_count,
	proto_get_child,
	proto_get_child_index,
	proto_add_child_at,
	proto_set_class,
	proto_get_in_flags,
	proto_set_in_flags,
	proto_get_name_count,
	proto_get_name,
	proto_set_name,
	proto_remove_name,
	proto_add_name,
	proto_check_node_flags,
	proto_set_node_flags,
	proto_reset_node_flags
};
static void proto_destroy(GF_ProtoInstance *inst)
{
	if (inst->nodes) gf_list_del(inst->nodes);
	if (inst->graph) {
#ifndef GPAC_DISABLE_LOG
		if (gf_log_tool_level(GF_LOG_DEBUG)) gf_log((inst->proto->name ? inst->proto->name : "Proto"), GF_LOG_INFO, "[Proto] Destroying graph for instance %p\n", inst);
#endif
		if (inst->graph->RootNode == (GF_Node*)inst) {
			gf_sg_del(inst->graph, (NodeID) gf_node_new(GF_NODE_ROOT));
		}
		gf_scene_graph_del(inst->graph);
	}
	if (inst->proto && inst->proto->instances) {
		u32 i;
		for (i=0; i<gf_list_count(inst->proto->instances); i++) {
			if ((GF_ProtoInstance *) gf_list_get(inst->proto->instances, i) == inst) {
				gf_list_rem(inst->proto->instances, i);
				break;
			}
		}
	}
	if (inst->params) gf_sg_vrml_field_reset(&inst->params, 1);
	gf_free(inst);
}
GF_EXPORT
void gf_proto_instance_del(GF_ProtoInstance *inst)
{
	if (!inst) return;
	/*if we're in the scene graph destroy the node*/
	if (inst->sgprivate.scenegraph) {
		gf_node_replace((NodeID) inst, NULL, 0);
		return;
	}

	proto_destroy(inst);
}
GF_EXPORT
void gf_scene_graph_proto_del(GF_SceneGraph *sg, NodeID pid)
{
	if (gf_node_get_tag(pid) != TAG_ProtoInstance) return;
	/*remove instance*/
	((GF_ProtoInstance *)pid)->sgprivate.scenegraph = NULL;
	proto_destroy(((GF_ProtoInstance *) pid));
	gf_node_replace(pid, NULL, 0);
}

static void proto_del_cbk(void *user)
{
	gf_scene_graph_del((GF_SceneGraph *) user);
}

GF_EXPORT
GF_ProtoInstance *gf_proto_instance_new(GF_Node *parent, GF_Proto *proto)
{
	u32 i;
	GF_ProtoInstance *inst = NULL;
	/*if we're not yet in the graph check for instance loop*/
	if (parent && !parent->sgprivate.scenegraph) {
		for (i=0; i<gf_list_count(proto->instances); i++) {
			inst = gf_list_get(proto->instances, i);
			if ((inst->parent == parent) && !strcmp(inst->name, proto->name)) return NULL;
		}
	}
	GF_SAFEALLOC(inst, GF_ProtoInstance);
	if (!inst) return NULL;
#ifndef GPAC_DISABLE_LOG
	if (gf_log_tool_level(GF_LOG_DEBUG)) gf_log((proto->name ? proto->name : "Proto"), GF_LOG_INFO, "[Proto] Creating instance %p for proto %s\n", inst, proto->name);
#endif
	inst->sgprivate.scenegraph = NULL;
	inst->sgprivate.parent = parent;
	/*if we're not in the graph but our parent is, we need to create a dummy scene graph*/
	if (parent && !parent->sgprivate.scenegraph) {
		GF_SceneGraph *sg = gf_scene_graph_new();
		gf_node_init(sg, parent);
		inst->sgprivate.scenegraph = sg;
		gf_node_register((NodeID) inst, proto->name ? proto->name : "");
		gf_scene_graph_del_node(sg, (NodeID) inst);
		/*and register our callback to destroy the dummy graph when we're destroyed*/
		inst->sgprivate.scenegraph = NULL;
		gf_node_set_callback_function((NodeID) inst, proto_del_cbk, sg);
	} else {
		/*otherwise set the parent to NULL so that we are not destroyed when exiting the graph*/
		inst->sgprivate.parent = NULL;
	}

	inst->proto = proto;
	if (proto) gf_list_add(proto->instances, inst);
	gf_node_init((GF_Node*)inst, parent ? parent : (GF_Node *) &proto_ifce);
	/*clone params*/
	for (i=0; i<gf_list_count(proto->fields.fieldArray); i++) {
		SFVrmlField *field = gf_list_get(proto->fields.fieldArray, i);
		SFVrmlField *new_field = gf_sg_vrml_mf_get_item(&inst->params, strdup(field->name), field->type);
		gf_sg_vrml_field_clone((NodeID) inst, new_field, field);
	}
	return inst;
}

GF_EXPORT
GF_ProtoInstance *gf_scene_graph_proto_new(GF_SceneGraph *sg, NodeID pid, GF_Proto *proto)
{
	u32 i;
	GF_Node *n = (GF_Node *) gf_node_get_private(pid);
	if (!n || !n->sgprivate.scenegraph || n->tag != TAG_ProtoInstance) return NULL;
	/*this is our graph, destroy old instance*/
	if (n->sgprivate.scenegraph == sg) {
		proto_destroy((GF_ProtoInstance *) pid);
	} else if (n->sgprivate.scenegraph && !sg) {
		/*detaching from graph, no need to create a new instance*/
		return (GF_ProtoInstance*)pid;
	}

	GF_SAFEALLOC(n, GF_ProtoInstance);
	if (!n) return NULL;
#ifndef GPAC_DISABLE_LOG
	if (gf_log_tool_level(GF_LOG_DEBUG)) gf_log((proto->name ? proto->name : "Proto"), GF_LOG_INFO, "[Proto] Creating instance %p for proto %s\n", n, proto->name);
#endif
	n->sgprivate.parent = NULL;
	/*clone params*/
	for (i=0; i<gf_list_count(proto->fields.fieldArray); i++) {
		SFVrmlField *field = gf_list_get(proto->fields.fieldArray, i);
		SFVrmlField *new_field = gf_sg_vrml_mf_get_item(&n->params, strdup(field->name), field->type);
		gf_sg_vrml_field_clone((NodeID) n, new_field, field);
	}
	/*setup instance*/
	n->tag = TAG_ProtoInstance;
	n->sgprivate.scenegraph = sg;
	n->sgprivate.parent = NULL;
	gf_node_set_callback_function((NodeID) n, proto_del_cbk, sg);
	n->sgprivate.UserCallback = gf_scene_graph_proto_del;
	n->sgprivate.name = strdup(proto->name);
	n->sgprivate.num_children = 0;
	n->sgprivate.flags = GF_SG_NODE_DIRTY | GF_SG_NODE_IS_ROOT;
	n->sgprivate.node_id = pid;
	gf_list_reset(n->sgprivate.children);
	/*clone proto*/
	n->proto = gf_scene_graph_clone_subgraph(sg, (NodeID) proto);
	if (!n->proto) {
		gf_node_unregister((NodeID) n, sg);
		return NULL;
	}
	/*setup proto instances*/
	for (i=0; i<gf_list_count(n->proto->instances); i++) {
		GF_ProtoInstance *p_inst = gf_list_get(n->proto->instances, i);
		if (!p_inst) continue;
		/*fix parent and proto in instance*/
		p_inst->sgprivate.parent = (GF_Node *) n;
		p_inst->proto = n->proto;
	}
	/*clone children*/
	for (i=0; i<gf_node_list_count(pid); i++) {
		NodeID ch_id = gf_node_list_get_child(pid, i);
		u32 j, count = gf_node_list_count(ch_id);
		GF_Node *ch = (GF_Node *) gf_node_get_private(ch_id);
		if (!ch) continue;
		/*clone the node*/
		ch = gf_scene_graph_clone_subgraph(sg, ch_id);
		if (!ch || !gf_node_register(ch, sg)) continue;
		ch->sgprivate.parent = n;
		/*update proto instances*/
		for (j=0; j<count; j++) {
			NodeID tn = gf_node_list_get_child(ch_id, j);
			GF_ProtoInstance *p_inst = (GF_ProtoInstance *) gf_node_get_private(tn);
			if (!p_inst) continue;
			/*fix parent and proto in instance*/
			p_inst->sgprivate.parent = ch;
			p_inst->proto = n->proto;
		}
		gf_list_add(n->sgprivate.children, ch);
		ch->sgprivate.node_id = (NodeID) ch;
		/*setup node params*/
		for (j=0; j<gf_list_count(n->proto->fields.fieldArray); j++) {
			SFVrmlField *f = gf_list_get(n->proto->fields.fieldArray, j);
			char *str = NULL;
			GF_SceneGraph *subgraph = (GF_SceneGraph *) ch->sgprivate.UserPrivate;
			if (!subgraph) subgraph = sg;
			switch (f->type) {
			case SFNode:
				if (strcmp(ch->sgprivate.name, f->name)) break;
				str = gf_sg_vrml_field_to_string(&n->params, f);
				gf_sg_set_sfnode_parameter(subgraph, str, (NodeID) ch);
				free(str);
				break;
			case MFNode:
			{
				u32 k, count = gf_sg_vrml_mf_get_count(&n->params, f);
				for (k=0; k<count; k++) {
					GF_Node *nch = gf_sg_vrml_mf_get_item(&n->params, f, k);
					if (!nch) continue;
					str = gf_sg_vrml_field_to_string(&n->params, f);
					gf_sg_set_mfnode_parameter(subgraph, str, (NodeID) ch);
				}
				free(str);
				break;
			}
			case SFString:
				if (!ch->sgprivate.UserString) break;
				str = gf_sg_vrml_field_to_string(&n->params, f);
				gf_sg_set_string_parameter(subgraph, ch->sgprivate.UserString, str);
				free(str);
				break;
			case MFString:
			{
				u32 k, count = gf_sg_vrml_mf_get_count(&n->params, f);
				for (k=0; k<count; k++) {
					GF_Node *nch = gf_sg_vrml_mf_get_item(&n->params, f, k);
					if (!nch) continue;
					str = gf_sg_vrml_field_to_string(&n->params, f);
					gf_sg_set_mfstring_parameter(subgraph, ch->sgprivate.UserString, str);
				}
				free(str);
				break;
			}
			default:
				break;
			}
		}
		n->sgprivate.num_children++;
	}
	return (GF_ProtoInstance *) n;
}